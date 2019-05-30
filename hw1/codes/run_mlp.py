from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 128, 0.01))
model.add(Sigmoid('active1'))
model.add(Linear('fc2', 128, 64, 0.01))
model.add(Sigmoid('active2'))
model.add(Linear('fc3', 64, 10, 0.01))
# loss = EuclideanLoss(name='loss')
loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 0.0001,
    'momentum': 0.001,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
}

test_loss, test_acc = [], []
train_loss, train_acc = [], []
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    trainloss, trainacc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    train_loss.extend(trainloss)
    train_acc.extend(trainacc)

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    testloss, testacc = test_net(model, loss, test_data, test_label, config['batch_size'])
    test_loss.append(testloss)
    test_acc.append(testacc)

plt.plot(train_loss,label='train loss vs. iterations',color='green')
plt.xlabel('iteration(s)')
plt.ylabel("train loss")
plt.title("train loss vs. iterations")
plt.show()
plt.plot(train_acc,label='train accuracy vs. iterations',color='r')
plt.xlabel('iteration(s)')
plt.ylabel("train accuracy")
plt.title("train accuracy vs. iterations")
plt.show()

plt.plot(test_loss,label='test loss vs. epochs',color='green')
plt.xlabel('epoch(s)')
plt.ylabel("test loss")
plt.title("test loss vs. epochs")
plt.show()
plt.plot(test_acc,label='test accuracy vs. epochs',color='r')
plt.xlabel('epoch(s)')
plt.ylabel("test accuracy")
plt.title("test accuracy vs. epochs")
plt.show()
