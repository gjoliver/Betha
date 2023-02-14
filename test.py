import ray
import torch.nn as nn
import torch.nn.functional as F


from mailman import fetch_tensor, save_tensor


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
        # First shard. Store result in Mailman.
        save_tensor("out1", self.blk2(self.blk1(x)))
        return None


@ray.remote
class TestLMShard2(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk3 = TestBlock(10, 10)
        self.out = nn.Linear(10, 10)

    def forward(self):
        out1 = fetch_tensor("out1")
        return F.softmax(self.out(self.blk3(out1)), dim=1)
