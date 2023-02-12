import torch.nn as nn
import torch.nn.functional as F


class TestBlock(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.layer = nn.Linear(*shape)

    def forward(self, x):
        return F.relu(self.layer(x))


class TestLM(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk1 = TestBlock(10, 10)
        self.blk2 = TestBlock(10, 10)
        self.blk3 = TestBlock(10, 10)
        self.out = nn.Linear(10, 10)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        output = F.softmax(self.out(x), dim=1)

        return output


# A manual sharding for testing.
# Notice that the last (output) module of the first shard "dropout1"
# is the input module of the second shard.
TEST_SHARDING = [
    {
        "inputs": [],
        "outputs": ["blk2"],
        "grad_inputs": ["blk3"],
        "grad_outputs": [],
        "modules": ["blk1", "blk2"],
    },
    {
        "inputs": ["blk2"],
        "outputs": [],
        "grad_inputs": [],
        "grad_outputs": ["blk3"],
        "modules": ["blk3", "out"],
    },
]
