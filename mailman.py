import ray


@ray.remote
class Mailman:
    def __init__(self):
        self._tensors = {}

    def get_tensor(self, key):
        return self._tensors[key]

    def save_tensor(self, key, tensor):
        self._tensors[key] = tensor

    def clear(self):
        self._tensors.clear()


def fetch_tensor(key):
    mailman = ray.get_actor("mailman")
    return ray.get(mailman.get_tensor.remote(key))


def save_tensor(key, tensor):
    mailman = ray.get_actor("mailman")
    ray.get([mailman.save_tensor.remote(key, tensor.to("cpu"))])
