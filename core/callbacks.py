import custom.callbacks


def get_callbacks(**kwargs):
    result = {}
    for k, v in kwargs.items():
        if hasattr(custom.callbacks, v):
            kwargs[k] = getattr(custom.callbacks, v)
            result[k] = kwargs[k]
        else:
            raise ValueError(f"没找到{k}这样的回调函数")

    return result
