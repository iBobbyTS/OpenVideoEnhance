from builder import Build
builder = Build(check_pytorch_version=False, build_type='install')
builder.DAIN(compute_compatibility=[70, 60])
