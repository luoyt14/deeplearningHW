import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from layer import BatchNorm2d, Reshape
from utils import LOG_INFO
import matplotlib.pyplot as plt
from myresnet import myResNet
from tensorboardX import SummaryWriter


writer = SummaryWriter(log_dir='../logs', comment='mnist-cnn')

TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
LOG_INTERVAL = 50
EPOCHS = 60

# TODO: adjust these hyperparameters
LR = 0.1
MM = 0.9
WD = 0.0001

device = 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
       transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

# model = nn.Sequential(
#     # TODO: implement your network architecture
#     nn.Conv2d(1, 4, 3, padding=1),
#     BatchNorm2d(4),
#     nn.ReLU(),
#     nn.AvgPool2d(2, stride=2),
#     nn.Conv2d(4, 4, 3, padding=1),
#     BatchNorm2d(4),
#     nn.ReLU(),
#     nn.AvgPool2d(2, stride=2),
#     Reshape((-1, 196)),
#     nn.Linear(196, 10)
# ).to(device)

model = myResNet().to(device)
input_data = torch.rand(16, 1, 28, 28)
with writer:
    writer.add_graph(model, (input_data,))

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MM, weight_decay=WD)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    acc_list = []

    train_loss_list = []
    train_acc_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        train_loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        train_acc_list.append(acc.item())
        if batch_idx % LOG_INTERVAL == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()
    return train_loss_list, train_acc_list


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


if __name__ == '__main__':
    test_loss, test_acc = [], []
    train_loss, train_acc = [], []

    for epoch in range(1, EPOCHS + 1):
        testloss, testacc = test(model, device, test_loader)
        test_loss.append(testloss)
        test_acc.append(testacc)

        trainloss, trainacc = train(model, device, train_loader, optimizer, epoch)
        train_loss.extend(trainloss)
        train_acc.extend(trainacc)

    plt.plot(train_loss, label='train loss vs. iterations', color='green')
    plt.xlabel('iteration(s)')
    plt.ylabel("train loss")
    plt.title("train loss vs. iterations")
    plt.show()
    plt.plot(train_acc, label='train accuracy vs. iterations', color='r')
    plt.xlabel('iteration(s)')
    plt.ylabel("train accuracy")
    plt.title("train accuracy vs. iterations")
    plt.show()

    plt.plot(test_loss, label='test loss vs. epochs', color='green')
    plt.xlabel('epoch(s)')
    plt.ylabel("test loss")
    plt.title("test loss vs. epochs")
    plt.show()
    plt.plot(test_acc, label='test accuracy vs. epochs', color='r')
    plt.xlabel('epoch(s)')
    plt.ylabel("test accuracy")
    plt.title("test accuracy vs. epochs")
    plt.show()
