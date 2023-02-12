import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from test import TestLM, TEST_SHARDING


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


def get_fake_output_hook(key):
    def hook(module, input, output):
        mailman = ray.get_actor("mailman")
        return ray.get(mailman.get_tensor.remote(key))
    return hook


def get_save_output_hook(key):
    def hook(module, input, output):
        mailman = ray.get_actor("mailman")
        ray.wait([mailman.save_tensor.remote(key, output)])
        # We do not have to change output here.
        return None
    return hook


class PatchedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = None
        self._out = None

    def _patch_model(self, sharding):
        assert sharding, "sharding should contain at least one module."

        for name, module in self._model.named_children():
            if name not in sharding["modules"]:
                # Replace model with Identity since these are the computations
                # that will actually run on a different instance.
                setattr(self._model, name, nn.Identity())

        # Handle input modules.
        for module_name in sharding["inputs"]:
            # These are the modules whose outputs are required
            # to run this shard.
            # We basically need to fake these outputs here.
            getattr(self._model, module_name).register_forward_hook(
                get_fake_output_hook(module_name)
            )

        # Handle output modules.
        for module_name in sharding["outputs"]:
            # These are the modules whose outputs are needed by
            # modules from other shards.
            # We must save them to the global mailman actor.
            getattr(self._model, module_name).register_forward_hook(
                get_save_output_hook(module_name)
            )

    def forward(self, t=None):
        if t is None:
            # Feed dummy data for non-first shards.
            t = torch.tensor(0)

        self._out = self._model(t)
        return self._out.detach().numpy()

    def load_and_patch(self, sharding):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class PatchedTestLM(PatchedModel):
    SHARDING_PLAN = TEST_SHARDING

    def load_and_patch(self, sharding):
        self._model = TestLM()
        self._patch_model(sharding)


@ray.remote(num_gpus=1)
class PatchedGPTJ6B(PatchedModel):
    SHARDING_PLAN = []

    def load_and_patch(self, sharding):
        self._model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        self._patch_model(sharding)
