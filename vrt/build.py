from builder import Build
builder = Build(check_pytorch_version=False, build_type='develop')
builder.BasicSR(cuda_extensions=False)
