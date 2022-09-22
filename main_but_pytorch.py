import pathlib

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision

transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(0.1307, 0.3081)])

train_data = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.MNIST('./datafiles/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden = nn.Linear(784, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = nn.functional.relu(x)
        return self.output(x)


print("loaded things")

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model_path = "mnist_savey.pth"

if not pathlib.Path(model_path).exists():
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.6f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))

dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
