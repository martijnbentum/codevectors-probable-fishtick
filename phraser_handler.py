from pathlib import Path
from stores import cgn

def _to_comp(filename):
    return Path(filename).parent.parent.name.split('-')[-1]

def load_phrases(min_dur = 3000, max_dur = 5000, filter_comps = 'acdhm'):
    '''load phrases from the database and filter them by duration and component.
    '''
    phrases = list(cgn.phrases.all())
    p = [x for x in phrases if x.duration > min_dur and x.duration < max_dur]
    phrases = [x for x in p if _to_comp(x.audio.filename) not in filter_comps]
    return phrases

def load_words(min_dur = 200, max_dur = 1500, filter_comps = 'acdhm'):
    '''load words from the database and filter them by duration and component.
    '''
    words = list(cgn.words.all())
    w = [x for x in words if x.duration > min_dur and x.duration < max_dur]
    words = [x for x in w if _to_comp(x.audio.filename) not in filter_comps]
    return words

def load_syllables(min_dur = 80, max_dur = 800, filter_comps = 'acdhm'):
    '''load syllables from the database and filter them by duration and component.
    '''
    syllables = list(cgn.syllables.all())
    s = [x for x in syllables if x.duration > min_dur and x.duration < max_dur]
    syllables = [x for x in s if _to_comp(x.audio.filename) not in filter_comps]
    return syllables

def load_phones(min_dur = 40, max_dur = 200, filter_comps = 'acdhm'):
    '''load phones from the database and filter them by duration and component.
    '''
    phones = list(cgn.phones.all())
    p = [x for x in phones if x.duration > min_dur and x.duration < max_dur]
    phones = [x for x in p if _to_comp(x.audio.filename) not in filter_comps]
    return phones

