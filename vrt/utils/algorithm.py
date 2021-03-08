import importlib

__all__ = (
    'get',
)

algorithms_belonging = {
    'ssm': 'vfin',
    'dain': 'vfin',
    'esrgan': 'sr',
    'edvr': 'sr'
}


def get(name):
    global ssm, dain, esrgan, edvr
    if name not in globals():
        globals()[name] = importlib.import_module(
            f'vrt.{algorithms_belonging[name]}.{name}.rter'
        ).RTer
    return globals()[name]
