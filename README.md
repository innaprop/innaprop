# INNAprop optimizer

This repository contains the PyTorch implementation of **INNAprop**, a second-order optimization algorithm for deep learning. INNAprop combines the Dynamical Inertial Newton (DIN) method with adaptive gradient scaling, inspired by RMSprop. The method leverages second-order geometric information while keeping computational costs and memory requirements comparable to AdamW.

Our paper, 'A second-order-like optimizer with adaptive gradient scaling for deep learning,' introduces INNAprop and demonstrates its performance on tasks such as image classification and large language model training, consistently matching or outperforming AdamW in speed and accuracy with minimal hyperparameter tuning.


## Usage

Here is a simple example of how to use INNAprop in a PyTorch training loop:

```
import torch
import torch.nn as nn
import torch.optim as optim
from innaprop import INNAprop

model = YourModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = INNAprop(model.parameters(), lr=1e-3, alpha=0.1, beta=0.9)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
```
