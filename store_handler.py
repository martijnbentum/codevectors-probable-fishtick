import json
from echoframe import store
from echoframe import batch_codebook_indices as bci
from phraser import models
from progressbar import progressbar
from pathlib import Path

import locations

def load_model_metadata(model_name):
    d = json.load(open(locations.data / 'model_paths.json'))
    for line in d:
        if line['model_name'] == model_name:
            return line

def wav2vec2_v1_model_names():
    d = json.load(open(locations.data / 'model_paths.json'))
    names = []
    for line in d:
        if 'wav2vec2_nl1' in line['model_name']:
            names.append(line['model_name'])
    return names


def make_or_load_store(model_name):
    store_root = locations.output_data / model_name
    s = store.Store(store_root)
    if s.load_model_metadata(model_name) is None:
        mmd = load_model_metadata(model_name)
        s.register_model(**mmd)
        print(f'new store for model {model_name} {mmd}')
    return s

def _to_comp(filename):
    return Path(filename).parent.parent.name.split('-')[-1]

def load_phrases():
    phrases = list(models.Phrase.objects.all())
    p = [x for x in phrases if x.duration > 3000 and x.duration < 5000]
    phrases = [x for x in p if _to_comp(x.audio.filename) not in 'acdhm']
    return phrases


def compute_codebook_indices(model_name, phrases, batch_size = 39):
    print(f'computing codebook indices for model {model_name}')
    print(f'batch size {batch_size}')
    print(f'number of phrases {len(phrases)}')
    store = make_or_load_store(model_name)
    print(f'store root {store.root}')
    bci.compute_codebook_indices_batch(phrases, model_name, store= store, 
        gpu = True, batch_size = batch_size)


def handle_model_names(model_names, phrases, batch_size = 39):
    for model_name in progressbar(model_names):
        compute_codebook_indices(model_name, phrases, batch_size = batch_size)
    




