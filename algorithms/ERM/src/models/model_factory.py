from algorithms.ERM.src.models.mnistnet import MNIST_CNN


nets_map = {"mnistnet": MNIST_CNN}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
