import ray
import torch.nn as nn
import torch.nn.functional as F


class TestBlock(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.layer = nn.Linear(*shape)

    def forward(self, x):
        return F.relu(self.layer(x))


@ray.remote
class TestLMShard1(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk1 = TestBlock(10, 10)
        self.blk2 = TestBlock(10, 10)

    def forward(self, x):
        return self.blk2(self.blk1(x))


@ray.remote
class TestLMShard2(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk3 = TestBlock(10, 10)
        self.out = nn.Linear(10, 10)

    def forward(self, x):
        return F.softmax(self.out(self.blk3(x)), dim=1)
