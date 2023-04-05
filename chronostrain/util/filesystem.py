from pathlib import Path
import glob
import math
from typing import List, Union
import hashlib


def convert_size(size_bytes: int) -> str:
    """
    Converts bytes to the nearest useful meaningful unit (B, KB, MB, GB, etc.)
    Code credit to https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python/14822210
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def files_in_dir(base_dir: str, extension: str = None) -> List[str]:
    """
    List all files in the specified directory, and filter by the specified extension (if applicable).
    :param base_dir: the directory to search.
    :param extension: if specified, filters the files by the extension. Example: "csv", "png", "txt", "pkl".
    :return: A list of path strings.
    """
    pattern = Path(base_dir) / "*" if extension is None else "*.{}".format(extension)
    return glob.glob(str(pattern))


def md5_checksum(file_path: Union[str, Path]):
    with open(file_path, "r") as f:
        return hashlib.md5(f.read().encode()).hexdigest()
