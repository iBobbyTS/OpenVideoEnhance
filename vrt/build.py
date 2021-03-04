import os
import sys

vrt_root = '/'.join(os.path.abspath(__file__).split('/')[:-2])
if os.getcwd() != vrt_root:
    os.chdir(vrt_root)
if vrt_root not in [os.path.abspath(path) for path in sys.path]:
    sys.path.append(vrt_root)


from builder import Build
builder = Build(download_model=False, build_type='develop', gitee=False)
builder.BasicSR()

# builder.SSM()
# builder.DAIN(build_type='install', cc=(60, 70))
