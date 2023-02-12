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

        
def move_tensor_to_device(tensor, device):
    if isinstance(tensor, list):
        return [
           move_tensor_to_device(e, device) for e in tensor           
        ]
    elif isinstance(tensor, dict):
        return {
            k: move_tensor_to_device(v, device)
            for k, v in tensor.items()
        }
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)


def get_mailman_read_hook(key):
    def hook(module, unused1, unused2):
        mailman = ray.get_actor("mailman")
        tensor = ray.get(mailman.get_tensor.remote(key))
        return move_tensor_to_device(tensor, "cuda:0")
    return hook


def get_mailman_write_hook(key):
    def hook(module, unused, output):
        mailman = ray.get_actor("mailman")
        output = move_tensor_to_device(output, "cpu")
        ray.get([mailman.save_tensor.remote(key, output)])
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

        for name, _ in self._model.named_children():
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

        # Now move everything to GPU.
        assert torch.cuda.is_available(), "No GPU?"
        for _, module in self._model.named_children():
            module.cuda()

        # Create optimizer now that everything is loaded.
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=0.1)

    def forward(self, data=None):
        if data is None:
            # Feed dummy data for non-first shards.
            data = torch.tensor(0)

        self._out = self._model(data.cuda())
        
        return self._out.detach().cpu().numpy()

    def backward(self, target=None):
        if target is None:
            # Dummy label.
            target = torch.randn(*self._out.shape)
        loss = F.mse_loss(self._out, target.cuda())
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
    SHARDING_PLAN = [
        {
            "inputs": [],
            "outputs": ["transformer.h.5"],
            "grad_inputs": ["transformer.h.6"],
            "grad_outputs": [],
            "modules": [
                "transformer.wte", "transformer.drop", "transformer.h.0", "transformer.h.1",
                "transformer.h.2", "transformer.h.3", "transformer.h.4", "transformer.h.5"
            ],
        },
        {
            "inputs": ["transformer.h.5"],
            "outputs": ["transformer.h.13"],
            "grad_inputs": ["transformer.h.14"],
            "grad_outputs": ["transformer.h.6"],
            "modules": [
                "transformer.h.6", "transformer.h.7", "transformer.h.8", "transformer.h.9",
                "transformer.h.10", "transformer.h.11", "transformer.h.12", "transformer.h.13"
            ],
        },
        {
            "inputs": ["transformer.h.13"],
            "outputs": ["transformer.h.21"],
            "grad_inputs": ["transformer.h.22"],
            "grad_outputs": ["transformer.h.14"],
            "modules": [
                "transformer.h.14", "transformer.h.15", "transformer.h.16", "transformer.h.17",
                "transformer.h.18", "transformer.h.19", "transformer.h.20", "transformer.h.21"
            ],
        },
        {
            "inputs": ["transformer.h.21"],
            "outputs": [],
            "grad_inputs": [],
            "grad_outputs": ["transformer.h.22"],
            "modules": [
                "transformer.h.22", "transformer.h.23", "transformer.h.24", "transformer.h.25",
                "transformer.h.26", "transformer.h.27", "transformer.ln_f", "lm_head"
            ],
        },
    ]

    def __init__(self, model_path):
        super().__init__()
        self._model_path = model_path

    def load_model(self):
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
        )
