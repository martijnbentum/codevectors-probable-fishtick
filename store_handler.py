import json
from echoframe import store
from echoframe import batch_codebook_indices as bci
from progressbar import progressbar
from ci_analysis import sort_w2v2_model_names

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
    names = sort_w2v2_model_names(names)
    return names

def make_or_load_store(model_name):
    store_root = locations.output_data / model_name
    s = store.Store(store_root)
    if s.load_model_metadata(model_name) is None:
        mmd = load_model_metadata(model_name)
        s.register_model(**mmd)
        print(f'new store for model {model_name} {mmd}')
    return s

    




