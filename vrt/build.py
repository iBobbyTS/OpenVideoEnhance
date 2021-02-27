from builder import Build
builder = Build(download_model=False, build_type='develop', gitee=True)
builder.BasicSR(rebuild=True)

# builder.SSM()
# builder.DAIN(build_type='install', cc=(60, 70))
