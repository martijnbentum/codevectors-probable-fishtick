import numpy as np


class CIStore:
    """Stores codebook index counts per model and phone class."""

    def __init__(self, models, phones, n_codes):
        self.models = list(models)
        self.phones = list(phones)
        self.n_codes = n_codes
        self.n_models = len(models)
        self.n_phones = len(phones)
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


def entropy(counts, base=2):
    '''Compute entropy from counts.
    counts:             iterable of non-negative counts
    base:               logarithm base (default bits)
    '''
    total = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return -(p * np.log(p) / np.log(base)).sum()

