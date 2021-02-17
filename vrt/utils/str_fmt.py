__all__ = (
    'file_size',
    'second2time'
)


def file_size(size):
    """
        Convert file size in bytes to human-readable format.

        Parameters
        ----------
        size : float
            File size in bytes.

        Returns
        -------
        file_size : str
            File size in a human-readable format.

        Examples
        --------
        >>> file_size(15481.8)
        15.12 KB
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return '%03.2f %sB' % (size, unit)
        size /= 1024.0
    return '%03.2f YB' % size


def second2time(second):
    """
        Convert file size in bytes to human-readable format.

        Parameters
        ----------
        second : float
            Time in seconds.

        Returns
        -------
        t : str
            Time in the format of (days:)hours:minutes:seconds.

        Examples
        --------
        >>> second2time(15481.8)
        4:18:01.80
    """
    if second < 60:
        return '%04.2fs' % second
    m, s = divmod(second, 60)
    if m < 60:
        return '%2d:%05.2f' % (m, s)
    h, m = divmod(m, 60)
    if h < 24:
        return '%2d:%02d:%05.2f' % (h, m, s)
    # More than 1 day
    d, h = divmod(h, 24)
    return '%d:%02d:%02d:%05.2f' % (d, h, m, s)
