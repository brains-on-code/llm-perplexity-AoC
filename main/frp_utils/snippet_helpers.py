from main.frp_utils.snippet_settings import SNIPPET_CLEAN, SNIPPET_OBF


def get_snippet_base(snippet: str) -> str:
    '''get snippet pair name'''
    if SNIPPET_CLEAN in snippet:
        return snippet.replace(SNIPPET_CLEAN, '')
    elif SNIPPET_OBF in snippet:
        return snippet.replace(SNIPPET_OBF, '')
    else:
        raise Exception(
            f'Snippet must contain either "{SNIPPET_CLEAN}" or "{SNIPPET_OBF}", none found in {snippet}')


def get_snippet_version(snippet: str) -> str:
    '''get snippet version'''
    assert ('-' in snippet), snippet
    version = snippet.split('-')[-1]
    assert len(version) == 2, version
    assert version[0] == 'v', version[0]
    assert version[1] in ['0', '1', '2'], version[1]
    return version


def get_snippet_variant(snippet: str) -> str:
    '''get snippet variant'''
    assert ('-' in snippet), snippet
    variant = snippet.split('-')[1]
    assert variant in [SNIPPET_CLEAN, SNIPPET_OBF], variant
    return variant


def get_snippet_number(snippet: str) -> str:
    '''get snippet number'''
    assert ('-' in snippet), snippet
    number = snippet.split('-')[0]
    if number == '':
        number = 0
    number = int(number)
    return number
