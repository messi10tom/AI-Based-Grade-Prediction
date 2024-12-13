import torch.nn as nn

# Define a simple regression model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 15)
        self.linear2 = nn.Linear(15, 10)
        self.linear3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x