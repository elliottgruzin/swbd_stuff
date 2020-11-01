#!/usr/bin/env python

import torch
from ffnn_dataset import Dataset
from feedforward import ExternalNetwork
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sys import argv

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': int(argv[1]),
          'shuffle': True,
          'num_workers': 4}
max_epochs = 100

# Datasets

print('******************* Network Info ********************\n')
print('Input Layer Size: 60\nBatch Size: {}\nNo. Layers: {}\nHidden Layer Width = {}\nLearning Rate: {}\nTraining Folder: {}'.format(argv[1],argv[2],argv[3],argv[4],argv[5]))
print('*****************************************************\n')


jar = open('data/{}/partition_dict.pkl'.format(argv[5]),'rb')
partition = pickle.load(jar)
jar2 = open('data/{}/label_dict.pkl'.format(argv[5]),'rb')
labels = pickle.load(jar2)
jar.close()
jar2.close()

# Generators

training_set = Dataset(partition['train'], labels, argv[5])
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['dev'], labels, argv[5])
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Models

ext_model = ExternalNetwork(20, int(argv[2]), int(argv[3]))
ext_model.cuda()
criterion = nn.BCELoss()
optimizer = optim.SGD(ext_model.parameters(), lr=float(argv[4]), momentum=0.9)

def test_loss():
    'Calculates loss from model on dev/test set'
    ext_model.eval()
    test_loss = 0
    for x, y in validation_generator:
        x_1 = x[:,:20]
        x_2 = x[:,60:80]
        x = torch.cat((x_1,x_2),1)
        x = x.to(device=torch.device('cuda:0'))
        y = y.to(device=torch.device('cuda:0'))
        pred_output = s(ext_model(x))
        loss = criterion(pred_output.float(), y.unsqueeze(1).float())
        # print(loss)
        test_loss += loss.data.cpu().numpy()
    return test_loss

num_epochs = 0
test_loss_list = []

s = nn.Sigmoid()

# Loop over epochs

for epoch in range(max_epochs):
    print('epoch {} done'.format(num_epochs))
    # Training
    total_loss = []
    ext_model.train()
    num_batches = 0
    for local_batch, local_labels in training_generator:
        # Concatenate first 20 elements from each speakers PFVA
        x_1 = local_batch[:,:20]
        x_2 = local_batch[:,60:80]
        x = torch.cat((x_1,x_2),1)
        num_batches += 1
        # Transfer to GPU
        local_batch, local_labels = x.to(device), local_labels.to(device)
        local_labels = local_labels.unsqueeze(1)
        # Model computations
        pred_labels = s(ext_model(local_batch))
        loss = criterion(pred_labels.float(), local_labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.data.cpu().numpy())

    print('mean loss in epoch ', epoch, sum(total_loss)/len(total_loss))
    t_loss = test_loss()
    print('test loss: ', t_loss)
    test_loss_list.append(t_loss)

    # Early stop

    if num_epochs > 10:
        test_loss_list.pop(0)
        if t_loss > test_loss_list[0]:
            # print('early stop at epoch', num_epochs)
            break
    torch.save(ext_model, 'ffnn.p')
    num_epochs += 1

# torch.save(ext_model, 'feedforward.p')
