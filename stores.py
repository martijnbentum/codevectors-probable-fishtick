from phraser import models
import locations

CGN_SOURCE_ID = 'cgn'
cgn = models.open_store(locations.cgn_lmdb)

def attach_cgn(echoframe_store):
    '''Attach the cgn phraser store as a registered source on an echoframe
    store. Required before saving segment-linked metadata.
    '''
    echoframe_store.attach_phraser_store(CGN_SOURCE_ID, cgn)
    return echoframe_store
