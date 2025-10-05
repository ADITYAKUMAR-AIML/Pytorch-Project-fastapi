import torch
import test_pytorch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # 2 inputs → 1 output

    def forward(self, x):
        return test_pytorch.sigmoid(self.fc(x))


if __name__ == "__main__":
    model = MyModel()
    torch.save(model.state_dict(), "model.pth")
