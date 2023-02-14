from typing import Callable, Dict

import ray
import torch


@ray.remote
class Shard:
    def __init__(self, module_creator: Callable, lr=0.1):
        self._module = module_creator()
        self._optimizer = torch.optim.AdamW(self._module.parameters(), lr=lr)

        # For forward and backward passes.
        self._inputs: Dict[str, torch.Tensor] = {}
        self._outputs: Dict[str, torch.Tensor] = {}

    def forward(self, inputs):
        self._inputs = inputs
        self._outputs = self._module(**inputs)

        return {
            k: v.detach() for k, v in self._outputs.items()
        }

    def backward(self, gradients):
        if not gradients:
            assert "loss" in self._outputs

            # Last layer. Backward with loss tensor.
            self._outputs["loss"].backward()
        else:
            # Non-last layer. Backward with gradients from next shard.
            assert set(self._outputs.keys()) == set(gradients.keys())

            # Run backward passes.
            for k, g in gradients.items():
                self._outputs[k].backward(gradient=g)

        # Return accumulated gradients on the input tensors.
        return {
            k: v.grad.data for k, v in self._inputs
        }

    def step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()
