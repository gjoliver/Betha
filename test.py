import torch.nn as nn
import torch.nn.functional as F


class TestBlock(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.layer = nn.Linear(*shape)

    def forward(self, x):
        return F.relu(self.layer(x))


class TestLMShard1(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk1 = TestBlock(10, 10)
        self.blk2 = TestBlock(10, 10)

    def forward(self, x):
        return {"x": self.blk2(self.blk1(x))}


class TestLMShard2(nn.Module):
    """A small toy model for testing purpose.
    """

    def __init__(self):
        super().__init__()

        self.blk3 = TestBlock(10, 10)
        self.out = nn.Linear(10, 10)

    def forward(self, x, labels = None):
        logits = F.softmax(self.out(self.blk3(x)), dim=1)
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        else:
            loss = None

        return {"loss": loss, "logits": logits}
