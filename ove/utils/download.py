import math

import requests
from tqdm import tqdm

from . import str_fmt

__all__ = (
    'set_proxy',
    'download_file_from_google_drive'
)


def set_proxy(ip='127.0.0.1', port=1086):
    """
        Setup socks5 proxy

        Parameters
        ----------
        ip : str
            IP address.
        port : int
            Port number between 1 and 65535
    """
    assert 1 <= port <= 65535
    import socket
    import socks

    socks.set_default_proxy(socks.SOCKS5, ip, port)
    socket.socket = socks.socksocket


def download_file_from_google_drive(file_id, save_path, chunk_size=32768):
    """
        Download file from google drive

        Parameters
        ----------
        file_id : str
            File ID of Google Drive shared file.
        save_path : str
            A path to save the file.
        chunk_size : int
            Chunk size of the downloading process.

        Returns
        -------
        finished : bool
            True if the download finished successfully, False otherwise.

        Examples
        --------
        >>> download_file_from_google_drive('1r-gVVu6oxCSZyBij4d4tPtssifGZlG5X', './destination.pth', chunk_size=131072)
        True
    """
    session = requests.Session()
    url = 'https://docs.google.com/uc?export=download'
    params = {'id': file_id}

    response = session.get(url, params=params, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params['confirm'] = value
    response = session.get(url, params=params, stream=True)

    # get file size
    response_file_size = session.get(
        url, params=params, stream=True, headers={'Range': 'bytes=0-2'})
    if 'Content-Range' in response_file_size.headers:
        file_size = int(
            response_file_size.headers['Content-Range'].split('/')[1])
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit='chunk')
        readable_file_size = str_fmt.file_size(file_size)
        downloaded_size = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size):
                downloaded_size += chunk_size
                pbar.update(1)
                pbar.set_description(f'Download {str_fmt.file_size(downloaded_size)} '
                                     f'/ {readable_file_size}')
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        pbar.close()
        return True
    else:
        return False
