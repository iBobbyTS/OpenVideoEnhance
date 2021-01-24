from builder import Build
builder = Build()
builder.DAIN(download_model=False, build_type='install')
builder.BasicSR(download_model=False)
