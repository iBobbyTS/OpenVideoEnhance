def dain():
    from vfin.dain.rter import rter
    return rter


def ssm():
    from vfin.ssm.rter import rter
    return rter


def bmbc():
    from vfin.bmbc.rter import rter
    return rter


def edvr():
    from sr.edvr.rter import rter
    return rter


def esrgan():
    from sr.esrgan.rter import rter
    return rter


def deoldify():
    from st.deoldify.rter import rter
    return rter


def bopbl():
    from st.bopbl.rter import rter
    return rter


modules = {
    'DAIN': [dain, 'vfin'],
    'SSM': [ssm, 'vfin'],
    'BMBC': [bmbc, 'vfin'],
    'EDVR': [edvr, 'sr'],
    'ESRGAN': [esrgan, 'sr'],
    'DeOldify': [deoldify, 'st'],
    'BOPBL': [bopbl, 'st']
}

def get(algorithm):
    assert algorithm in modules.keys(), f'{algorithm} not found in supported algorithms: {modules.keys()}'
    return modules[algorithm][0]()

def belong_to(algorithm):
    assert algorithm in modules.keys(), f'{algorithm} not found in supported algorithms: {modules.keys()}'
    return modules[algorithm][1]