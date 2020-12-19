def second2time(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    t = '%d:%02d:%05.2f' % (h, m, s)
    return t
