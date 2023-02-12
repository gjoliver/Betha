import torch
import torch.nn as nn
import torch.nn.functional as F


class TestLM(nn.Module):
    """A small toy model for testing purpose.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


# A manual sharding for testing.
# Notice that the last (output) module of the first shard "dropout1"
# is the input module of the second shard.
TEST_SHARDING = [
    {
        "inputs": [],
        "outputs": ["dropout1"],
        "modules": ["conv1", "conv2", "dropout1"],
    },
    {
        "inputs": ["dropout1"],
        "outputs": [],
        "modules": ["dropout2", "fc1", "fc2"],
    },
]
