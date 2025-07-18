import os

import mne
import numpy as np

mne.set_log_level('WARNING')


class EEGDataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_eeg_file_path(self, participant_id, trial, extension):
        return os.path.join(self.base_dir, participant_id, "EEG", "EEG_FRP", f"EEG_FRP_trial_{trial}.{extension}")

    def get_trials(self, participant_id):
        eeg_folder = os.path.join(self.base_dir, participant_id, "EEG", "EEG_FRP")
        return [f.split("_")[-1].split(".")[0] for f in os.listdir(eeg_folder) if f.endswith(".eeg")] if os.path.exists(
            eeg_folder) else []

    # ---------------------- EEG Data Loading ---------------------- #
    def load_eeg_data(self, participant_id, trial):
        eeg_vhdr_path = self.get_eeg_file_path(participant_id, trial, "vhdr")
        if not os.path.exists(eeg_vhdr_path):
            print(f"Warning: Missing EEG file for Participant {participant_id}, Trial {trial}")
            return None, None
        eeg_data = mne.io.read_raw_brainvision(eeg_vhdr_path, preload=True, verbose=None)
        events, _ = mne.events_from_annotations(eeg_data, verbose=None)
        return eeg_data, events

    # ---------------------- EEG Epoch Extraction ---------------------- #
    def extract_epochs(self, eeg_data, events, event_id, tmin=-0.3, tmax=1.0, channel=None):
        matching_event_id = {key: val for key, val in event_id.items() if val in events[:, -1]}
        if not matching_event_id:
            print("No matching events found.")
            return None
        # print matching evend id
        print(f"Matching Event ID: {matching_event_id}")
        epochs = mne.Epochs(eeg_data, events, event_id=matching_event_id, tmin=tmin,
                            tmax=tmax, baseline=None, preload=True, verbose=None, event_repeated='merge')

        if channel:
            try:
                epochs = epochs.pick_channels([channel])
            except:
                print(f"Warning: Epoch does not contain any valid data for channel {channel}.")
                return None

        return epochs

    def load_all_participants_individual_snippets(self, participant_ids=None, t_min=-0.3, t_max=1.0, channel=None,
                                                  downsample_freq=None):
        """
        Loads EEG data for all participants, extracts mean potentials as features,
        and returns feature matrices (X) and labels (y).
        """
        X, y, groups, trial_ids = [], [], [], []
        event_id = {'AOI 1': 101, 'AOI 2': 102, 'AOI 3': 103, 'AOI 4': 104, 'AOI 5': 105, 'AOI 6': 106, 'AOI 7': 107,
                    'AOI 8': 108, 'AOI 9': 109}

        if not participant_ids:
            participant_ids = [folder for folder in os.listdir(
                self.base_dir) if os.path.isdir(os.path.join(self.base_dir, folder))]
            participant_ids.remove("total")  # Exclude aggregate data folders

        for participant_id in participant_ids:
            trials = self.get_trials(participant_id)
            for trial in trials:
                eeg_data, events = self.load_eeg_data(participant_id, trial)
                if eeg_data is None or events is None:
                    continue

                epochs = self.extract_epochs(eeg_data, events, event_id,
                                             tmin=t_min, tmax=t_max, channel=channel)
                if epochs is None or len(epochs) == 0:
                    print(f"Skipping Trial {trial} for Participant {participant_id}: No expected events found.")
                    continue
                if downsample_freq:
                    epochs.resample(downsample_freq, npad="auto")

                # Filter epochs to only those with valid event codes
                valid_mask = np.isin(epochs.events[:, -1], list(event_id.values()))
                if not np.any(valid_mask):
                    print(f"Skipping Trial {trial} for Participant {participant_id}: No valid AOI events found.")
                    continue

                eeg_array = epochs.get_data()[valid_mask]
                filtered_events = epochs.events[valid_mask]

                labels = []
                trial_id_list = []
                for event_code in filtered_events[:, -1]:
                    for key, value in event_id.items():
                        if value == event_code:
                            labels.append(key)
                            trial_id_list.append(f"{trial}-{key}")
                            break

                X.append(eeg_array)
                y.append(labels)
                groups.append(np.full(len(labels), participant_id))
                trial_ids.append(np.array(trial_id_list))
        if not X:
            raise ValueError("No valid EEG data found. Check event markers and trial structure.")

        X = np.vstack(X)
        y = np.concatenate(y)
        groups = np.concatenate(groups)
        trial_ids = np.concatenate(trial_ids)

        return X, y, groups, trial_ids

    def load_all_participants(self, participant_ids=None, t_min=-0.3, t_max=1.0, channel=None, downsample_freq=None):
        """
        Loads EEG data for all participants, extracts mean potentials as features,
        and returns feature matrices (X) and labels (y).
        """
        X, y, groups, trial_ids = [], [], [], []
        event_id = {"confusing": 172, "clean": 171}  # Event markers

        if not participant_ids:
            participant_ids = [folder for folder in os.listdir(
                self.base_dir) if os.path.isdir(os.path.join(self.base_dir, folder))]
            participant_ids.remove("total")  # Exclude aggregate data folders

        for participant_id in participant_ids:
            trials = self.get_trials(participant_id)
            for trial in trials:
                eeg_data, events = self.load_eeg_data(participant_id, trial)
                if eeg_data is None or events is None:
                    continue  # Skip if data is missing

                epochs = self.extract_epochs(eeg_data, events, event_id,
                                             tmin=t_min, tmax=t_max, channel=channel)
                if epochs is None or len(epochs) == 0:
                    print(
                        f"Skipping Trial {trial} for Participant {participant_id}: No expected events found.")
                    continue
                if downsample_freq:
                    epochs.resample(downsample_freq, npad="auto")

                eeg_array = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
                labels = np.array([1 if 172 in epochs.events[:, -1] else 0] * len(epochs))

                X.append(eeg_array)
                y.append(labels)
                groups.append(np.full(len(labels), participant_id))
                trial_ids.append(np.full(len(labels), trial))

        if not X:
            raise ValueError("No valid EEG data found. Check event markers and trial structure.")

        X = np.vstack(X)
        y = np.concatenate(y)
        groups = np.concatenate(groups)
        trial_ids = np.concatenate(trial_ids)

        return X, y, groups, trial_ids

    @staticmethod
    def average_trials(X, trial_ids):
        """
        Average the EEG signals across epochs belonging to the same trial and compute the standard deviation.
        
        Parameters
        ----------
        X : numpy.ndarray
            EEG data with shape (n_epochs, n_channels, n_times).
        trial_ids : numpy.ndarray
            Array of trial IDs corresponding to each epoch.
        
        Returns
        -------
        averaged_signals : dict
            Dictionary where keys are trial IDs and values are the averaged EEG signals.
        std_signals : dict
            Dictionary where keys are trial IDs and values are the standard deviation of EEG signals.
        """
        unique_trials = np.unique(trial_ids)
        averaged_signals = {}
        std_signals = {}
        for trial in unique_trials:
            idx = np.where(trial_ids == trial)[0]
            avg_signal = np.mean(X[idx], axis=0)
            std_signal = np.std(X[idx], axis=0)
            averaged_signals[trial] = avg_signal
            std_signals[trial] = std_signal

        return averaged_signals, std_signals

    @staticmethod
    def max_trials(X, trial_ids):
        """
        Calculate the maximum EEG signals across epochs belonging to the same trial and compute the standard deviation.
        
        Parameters
        ----------
        X : numpy.ndarray
            EEG data with shape (n_epochs, n_channels, n_times).
        trial_ids : numpy.ndarray
            Array of trial IDs corresponding to each epoch.
            
        Returns
        -------
        max_signals : dict
            Dictionary where keys are trial IDs and values are the maximum EEG signals.
        std_signals : dict
            Dictionary where keys are trial IDs and values are the standard deviation of EEG signals.
        """
        unique_trials = np.unique(trial_ids)
        max_signals = {}

        for trial in unique_trials:
            idx = np.where(trial_ids == trial)[0]
            # Calculate maximum value at each time point across epochs
            max_signal = np.max(X[idx], axis=0)
            max_signals[trial] = max_signal

        return max_signals
