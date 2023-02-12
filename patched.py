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


def get_mailman_read_hook(key):
    def hook(module, input, output):
        mailman = ray.get_actor("mailman")
        return ray.get(mailman.get_tensor.remote(key))
    return hook


def get_mailman_write_hook(key):
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
        self._optimizer = None
        self._out = None

    def prepare(self, sharding):
        assert self._model is None, "prepare() can only be called once."
        assert sharding, "sharding should contain at least one module."

        self.load_model()

        for name, module in self._model.named_children():
            if name not in sharding["modules"]:
                # Replace model with Identity since these are the computations
                # that will actually run on a different instance.
                setattr(self._model, name, nn.Identity())

        # Handle input modules.
        for name in sharding["inputs"]:
            # These are the modules whose outputs are required
            # to run this shard.
            # We need to fetch and fake these outputs here.
            getattr(self._model, name).register_forward_hook(
                get_mailman_read_hook(name)
            )

        # Handle output modules.
        for name in sharding["outputs"]:
            # These are the modules whose outputs are needed by
            # modules from other shards.
            # We must save them to the global mailman actor.
            getattr(self._model, name).register_forward_hook(
                get_mailman_write_hook(name)
            )

        # Handle gradient input modules.
        for name in sharding["grad_inputs"]:
            # Fetch gradients to be flown into the upstream modules.
            getattr(self._model, name).register_full_backward_hook(
                get_mailman_read_hook(f"{name}/grads")
            )

        # Handle gradient output modules.
        for name in sharding["grad_outputs"]:
            # Fetch gradients to be flown into the upstream modules.
            getattr(self._model, name).register_full_backward_hook(
                get_mailman_write_hook(f"{name}/grads")
            )

        # Create optimizer now that everything is loaded.
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=0.1)

    def forward(self, data=None):
        if data is None:
            # Feed dummy data for non-first shards.
            data = torch.tensor(0)

        self._out = self._model(data)
        return self._out.detach().numpy()

    def backward(self, target=None):
        if target is None:
            # Dummy label.
            target = torch.randn(*self._out.shape)
        loss = F.mse_loss(self._out, target)
        loss.backward()
        
        self._out = None

    def step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

    def load_model(self, sharding):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class PatchedTestLM(PatchedModel):
    SHARDING_PLAN = TEST_SHARDING

    def load_model(self):
        self._model = TestLM()


@ray.remote(num_gpus=1)
class PatchedGPTJ6B(PatchedModel):
    SHARDING_PLAN = []

    def load_model(self):
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )
