import os
import time

from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.parent


def get_data_folder():
    return os.path.join(get_project_root(), 'data')


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time() - ts
        print(f'{te} sec')
        return result
    return timed
