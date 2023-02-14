# Betha

Betha is a tribute to [Alpa](https://github.com/alpa-projects/alpa)

While Alpha is the nuclear weapon for large model training and serving, Betha is a toy example of model parallel training of a simple GPT-J implementation using manual sharding and Ray core.

[model.py](https://github.com/gjoliver/Betha/blob/master/model.py) has the broken up GPT-J model. [shard.py](https://github.com/gjoliver/Betha/blob/master/shard.py) is the Ray actor wrapper that helps flow tensors and gradients across nodes.

To run: ``python train --model_dir=<cached HF GPT-J model>``
