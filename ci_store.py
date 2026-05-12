import json
import numpy as np
from pathlib import Path


class CIStore:
    """Stores codebook index counts per model and phone class."""

    def __init__(self, models, phones, n_codes, directory, name = 'ci_counts'):
        self.models = list(models)
        self.phones = list(phones)
        self.n_codes = n_codes
        self.directory = Path(directory)
        self.name = name
        self.n_models = len(self.models)
        self.n_phones = len(self.phones)
        self._model_idx = {m: i for i, m in enumerate(self.models)}
        self._phone_idx = {p: i for i, p in enumerate(self.phones)}
        c = np.zeros((self.n_models, self.n_phones, n_codes), dtype=np.int64)
        self.counts = c
        
    def __repr__(self):
        m = f'CIStore(models={self.n_models}, '
        m += f'phones={self.n_phones}, '
        m += f'n_codes={self.n_codes})'
        return m

    def add(self, model, phone, code_indices):
        """Increment counts for the given model/phone and one or more code indices."""
        m = self._model_idx[model]
        p = self._phone_idx[phone]
        np.add.at(self.counts[m, p], code_indices, 1)

    def get(self, model=None, phone=None, code=None):
        """
        Return a count or slice. 
        Omit an argument to get all values along that axis.
        All three specified → scalar int.
        Two specified → 1-D array.
        One specified → 2-D array.
        None specified → full 3-D array.
        """
        m = self._model_idx[model] if model is not None else slice(None)
        p = self._phone_idx[phone] if phone is not None else slice(None)
        c = code if code is not None else slice(None)
        return self.counts[m, p, c]

    def model_phone_entropy(self, model, phone, base=2):
        '''Compute the entropy of the codebook index distribution 
        for a given model and phone.
        '''
        counts = self.get(model=model, phone=phone)
        return entropy(counts, base=base)
    
    def model_entropy(self, model, base=2):
        '''Compute the entropy of the codebook index distribution 
        for a given model.
        '''
        counts = self.get(model=model)
        counts = counts.sum(axis=0) 
        return entropy(counts, base=base)
    

    def save(self, overwrite = False):
        self.directory.mkdir(parents=True, exist_ok=True)
        npy_filename = self.directory / f'{self.name}.npy'
        json_filename = self.directory / f'{self.name}.json'
        if not overwrite and (npy_filename.exists() or json_filename.exists()):
            m = f'files {npy_filename} or {json_filename} already exist'
            raise FileExistsError(m)
        np.save(npy_filename, self.counts)
        d = {'models': self.models, 'phones': self.phones, 'n_codes': self.n_codes}
        with open(json_filename, 'w') as fout:
            json.dump(d, fout)

    @classmethod
    def load(cls, directory, name = 'ci_counts'):
        directory = Path(directory)
        npy_filename = directory / f'{name}.npy'
        json_filename = directory / f'{name}.json'
        if not (npy_filename.exists() and json_filename.exists()):
            m = f'files {npy_filename} or {json_filename} do not exist'
            raise FileExistsError(m)
        with open(json_filename, 'r') as fin:
            d = json.load(fin)
        store = cls(d['models'], d['phones'], d['n_codes'], directory, name)
        store.counts = np.load(npy_filename)
        store.validate_counts_shape()
        return store

    def validate_counts_shape(self):
        expected = (self.n_models, self.n_phones, self.n_codes)
        if self.counts.shape != expected:
            m = f'counts shape {self.counts.shape} does not match metadata '
            m += f'{expected}'
            raise ValueError(m)



        
