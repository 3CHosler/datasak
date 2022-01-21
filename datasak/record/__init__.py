#Brian Hosler
#January 2022


from .. import backend

pack=None
unpack=None
fmt=None

def set_recordFormat(feat=backend.featType):
    '''
    By scoping the imports within this function, they aren't exported
    as part of the features module
    '''
    import importlib
    from importlib import import_module
    global pack, unpack, feature
    if feat=='patch' or None:
        fmt=import_module('.patch',package='rename.record')
    elif feat=='qtab':
        fmt=import_module('.qtab',package='rename.record')
    else:
        spec = importlib.util.spec_from_file_location('recordFormat', feat)
        fmt = importlib.util.module_from_spec(spec)
        sys.modules['recordFormat'] = fmt
        spec.loader.exec_module(fmt)
    pack = fmt.pack
    unpack = fmt.unpack

set_recordFormat()

del backend
