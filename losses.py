import torch
import torch.nn as nn

bce_loss = nn.BCELoss(weight=torch.tensor([10]))

if __name__ == "__main__":
    m = nn.Sigmoid()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = bce_loss(m(input), target)
    print(output)