import importlib

__all__ = (
    'get',
)


def get(name):
    global ssm, dain, esrgan, edvr
    if name not in globals():
        globals()[name] = importlib.import_module(
            f'vrt.algorithm.{name}.rter'
        ).RTer
    return globals()[name]
