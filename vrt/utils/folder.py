import os


__all__ = (
    'listdir',
    'check_model',
    'path2list',
    'check_dir_availability',
)


def path2list(path):
    """
        Convert file path to the mother path, file name and extension.

        Parameters
        ----------
        path : str
            File path

        Returns
        -------
        file_size : list
            List of the mother path, file name and extension.

        Examples
        --------
        >>> path2list('/content/videos/test.mov')
        ['/content/videos', 'test', '.mov']
    """
    path, name = os.path.split(path)
    name, ext = os.path.splitext(name)
    return [path, name, ext]


def listdir(folder) -> list:  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    """
        List the objects in the input directory, remove unwanted ones and sort them.

        Parameters
        ----------
        folder : str
            File path

        Returns
        -------
        files : list
            List of the files in the input folder.

        Examples
        --------
        >>> listdir('/content/videos')
        ['av1.mp4', 'raw.yuv', 'test.mov']
    """
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = []
    for file in os.listdir(folder):
        if file not in disallow and file[:2] != '._':
            files.append(file)
    files.sort()
    return files


def check_model(model_dir_folder, path, default_path):
    """
        Check if models exist.

        Parameters
        ----------
        paths : list or str
            File path
    """
    if path is None:
        path = os.path.abspath(os.path.join(
            model_dir_folder, default_path
        ))
    while not os.path.exists(path):
        input(f"{path} not found, click return when the file is ready")
    return path


def check_dir_availability(dire, ext=''):
    """
        Check availability of the specified directory, add number after file name if exists.

        Parameters
        ----------
        dire : str
            Output path without extension.
        ext : str
            Extension of the output file.

        Returns
        -------
        dire : str
            Fixed directory without existance problems.

        Examples
        --------
        >>> check_dir_availability('/content/videos/test', '.mov')
        '/content/videos/test_2.mov'
    """
    dire = os.path.abspath(dire)
    if ext:  # is file
        if ext[0] != '.':
            ext = '.' + ext
        os.makedirs(os.path.split(dire)[0], exist_ok=True)
    if os.path.exists(dire + ext):  # If target file/folder exists
        count = 2
        while os.path.exists(f'{dire}_{count}{ext}'):
            count += 1
        dire = f'{dire}_{count}{ext}'
    else:
        dire = f'{dire}{ext}'
    if not ext:  # Output as folder
        os.makedirs(dire)
    return dire
