import torch

class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)