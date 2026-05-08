import store_handler as sh
from echoframe import batch_codebook_indices as bci

def handle_model_names(model_names, items = None, gpu = True, batch_size = 39):
    '''compute codebook indices for a list of model names and items.
    model_names: list of model names to compute codebook indices for
    items: iterable of phraser segments (phrase by default)
    '''
    if items is None:
        import phraser_handler
        items = phraser_handler.load_phrases()
    for model_name in progressbar(model_names):
        compute_codebook_indices(model_name, items, batch_size = batch_size)

def compute_codebook_indices(model_name, items, gpu = True, batch_size = 39):
    '''compute codebook indices for a given model name and items.
    model_name: name of the model to compute codebook indices for
    items: iterable of phraser segments (phrase by default)
    '''
    print(f'computing codebook indices for model {model_name}')
    print(f'batch size {batch_size}')
    print(f'number of items {len(items)}')
    store = make_or_load_store(model_name)
    print(f'store root {store.root}')
    bci.compute_codebook_indices_batch(items, model_name, store= store, 
        gpu = gpu, batch_size = batch_size)

