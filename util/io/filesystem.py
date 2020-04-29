import os
import math

def convert_size(size_bytes):
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


def get_filesize_bytes(filename) -> int:
    """
    Get the size of the specified file, in bytes. Use convert_size() for a more meaningful inference_output.
    """
    return os.stat(filename).st_size
