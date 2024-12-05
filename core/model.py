import custom.model


def get_model(name, **kwargs):
    return getattr(custom.model, name)(**kwargs)

