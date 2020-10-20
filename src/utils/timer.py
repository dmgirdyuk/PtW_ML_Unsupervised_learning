import time


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time() - ts
        print(f'{te} sec')
        return result
    return timed
