import custom.model


def get_model(name, **kargs):
    return getattr(custom.model, name)(**kargs)

