import store_handler
import phraser_handler
import frame
import ci_store
from progressbar import progressbar

def handle_model_name(model_name, counts, flatten_ci = True):
    store = load_store(model_name)
    print(f'handling model {model_name}')
    error = []
    for metadata in progressbar(store.metadatas):
        try: handle_metadata(metadata, counts, flatten_ci)
        except ValueError as e: error.append((metadata, str(e)))
        except AttributeError as e: error.append((metadata, str(e)))
    return error
    

def handle_metadata(metadata, counts, flatten_ci = True):
    phrase = metadata.phraser_object
    frames = phrase_to_frames(phrase)
    phrase_ci = metadata.load_payload()
    if frames.n_frames != phrase_ci.shape[0]:
        m = f'n frames {frames.n_frames} does not match codebook indices '
        m += f'{phrase_ci.shape[0]}, for phrase {phrase}'
        m += f' and {metadata} and model {metadata.model_name}'
        raise ValueError(m)
    skipped_phones = set() 
    for phone in phrase.phones:
        if phone.label not in counts.phones:
            skipped_phones.add(phone.label)
            continue
        phone_ci= handle_phone(phone, frames, phrase_ci)
        if flatten_ci: phone_ci = flatten(phone_ci) 
        counts.add(metadata.model_name, phone.label, phone_ci)
    if skipped_phones:
        print(f'skipped phones for phrase {phrase}: {skipped_phones}')
    
def handle_phone(phone, frames, codebook_indices):    
    selected_frames = phone_to_selected_frames(phone, frames)
    indices = [x.index for x in selected_frames]
    selected_codebook_indices = codebook_indices[indices]
    return selected_codebook_indices

def load_store(model_name):
    return store_handler.make_or_load_store(model_name)

def phrase_to_frames(phrase):
    duration = phrase.duration_seconds()
    start = phrase.start_seconds
    f = frame.make_frames_from_duration(duration, start_time = start)
    return f

def phone_to_selected_frames(phone, frames, overlap = 100):
    start = phone.start_seconds
    end = phone.end_seconds
    selected_frames = frames.select_frames(start, end, overlap)
    return selected_frames

def flatten(items):
    '''Flatten a list of lists.
    items:              iterable containing iterables
    '''
    return [x for sublist in items for x in sublist]

def ci_tuple_to_index(ci_tuple):
    return ci_tuple[0] * 320 + (ci_tuple[1] - 320)

def load_phone_labels():
    with open('../data/phone_labels','r') as fin:
        labels = fin.read().split('\n')
    return labels

def make_counts(model_names = None, phone_labels = None, n_codes = 640, 
    directory = '../ci_stores'):
    return ci_store.CIStore(model_names, phone_labels, n_codes, directory)


