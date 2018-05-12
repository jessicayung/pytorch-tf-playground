"""
Logistic Regression model
Caveat: this was coded to see how you can build a simple model in PyTorch.
It's a pretty bad model for the proposed problem.
Adapted from https://hsaghir.github.io/data_science/pytorch_starter/

May 2018
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionClassifier, self).__init__()
        # Create linear layer and define params of layer
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_vector):
        return nn.functional.softmax(self.linear(input_vector))

# create data
num_features = 1
num_datapoints = 20
x = torch.randn(num_datapoints, num_features)
y = x.sum(1) > 0
# minibatch_size = 5
# x = x.view(-1, minibatch_size, 10)
# y = y.view(-1, minibatch_size)
y2 = torch.zeros(num_datapoints, 2)
y2[np.arange(num_datapoints), np.array(y)] = 1
y = y2
data = zip(x, y)

print("Data:", x, y)

# create model
model = LogisticRegressionClassifier(num_features, 2)

# define loss functino
loss = nn.MSELoss()

# optimiser
optimiser = optim.SGD(model.parameters(), lr=1e-1)


# train for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    for xpt, ypt in data:
        # reset gradients, else pytorch accumulates them
        model.zero_grad()
        print("x:",xpt)

        # forward pass
        yhat = model(autograd.Variable(xpt))
        print("yhat: ", yhat)

        # backward pass
        L = loss(yhat, ypt)
        print("Loss: ", L)
        # backprop (calculate gradients)
        L.backward()
        # update parameters
        optimiser.step()
        print("Updated parameters:", model.parameters())




