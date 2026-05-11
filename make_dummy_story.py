import ci_store


def make_dummy_ci_store(models = None, phones = None, n_codes = 8,
    directory = '.', name = 'dummy_ci_counts', n_items = 100):
    if models is None: models = ['model_a', 'model_b']
    if phones is None: phones = ['aa', 'ee', 'oo']
    store = ci_store.CIStore(models, phones, n_codes, directory, name)
    add_dummy_counts(store, n_items)
    return store

def add_dummy_counts(store, n_items = 100):
    for model_index, model in enumerate(store.models):
        for phone_index, phone in enumerate(store.phones):
            codes = make_dummy_codes(model_index, phone_index, store.n_codes, n_items)
            store.add(model, phone, codes)

def make_dummy_codes(model_index, phone_index, n_codes, n_items):
    offset = model_index + phone_index
    return [(i + offset) % n_codes for i in range(n_items)]

def make_and_save(directory = '.', name = 'dummy_ci_counts', overwrite = False):
    store = make_dummy_ci_store(directory = directory, name = name)
    store.save(overwrite = overwrite)
    return store


if __name__ == '__main__':
    make_and_save(overwrite = True)
