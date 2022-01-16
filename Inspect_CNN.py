# coding=utf-8
# Written and Edited by Mingxuan Zhang
# 2020-Nov

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

global gradient

gradient=[]

# torch.cuda.is_available()
# device = torch.device("cuda:0")

# ----enable cuda if possible----#
def device_detect():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device

# ----prepare data----#
def data_preprocessing():
    training_data = np.load("training_data.npy",  allow_pickle=True)

    X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])
    Validation_PCT = 0.1
    val_size = int(len(X)*Validation_PCT)

    # ----transfer data to GPU if you have one, otherwise to CPU----#
    if torch.cuda.is_available():
        Train_X = X[:-val_size].cuda()
        Train_y = y[:-val_size].cuda()

        test_X = X[-val_size:].cuda()
        test_y = y[-val_size:].cuda()
    else:
        Train_X = X[:-val_size]
        Train_y = y[:-val_size]

        test_X = X[-val_size:]
        test_y = y[-val_size:]

    return Train_X, Train_y, test_X, test_y

# ----define network----#
class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.conv2 = nn.Conv2d(2, 2, 2)
        self.conv3 = nn.Conv2d(2, 2, 2)
        self._to_linear = None
        self.middleoutputs = []
        self.convrealresults = []
        self.gradient = []

        x = torch.randn(50,50).view(-1,1,50,50)
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512,2)

    def inside_visulization(self,layer_output):
        # ----observe feature maps in hidden layers----#
        x2 = layer_output.detach().numpy()
        X3 = x2[0][0]
        plt.imshow(X3)
        plt.show()
        return X3

    def convs(self, x):
        # self.inside_visulization(x)
        # max pooling over 2x2
        # input = self.inside_visulization(x) #middle_layers0, input
        # self.middleoutputs.append(x)

        x = self.conv1(x) #49x49
        # self.convrealresults.append(x) # first Convolution results

        x = F.relu(x)
        # self.middleoutputs.append(x)
        x = F.max_pool2d(x, (2, 2)) #24x24
        # self.convrealresults.append(x) # input before the second Convolution opreation
        # x.register_hook(hook_function)

        x = self.conv2(x) # 23x23
        # x.register_hook(hook_function)
        # self.convrealresults.append(x) # second Convolution results
        x = F.relu(x)
        # self.convrealresults.append(x)
        # x.register_hook(hook_function)

        x = F.max_pool2d(x, (2, 2)) #11x11
        self.convrealresults.append(x)
        x.register_hook(hook_function)

        # self.middleoutputs.append(x)
        x = self.conv3(x) # 10x10
        # self.convrealresults.append(x)
        x.register_hook(hook_function)

        x = F.relu(x)
        # x.register_hook(hook_function)

        x = F.max_pool2d(x, (2,2)) # 5x5
        # x.register_hook(hook_function)

        if self._to_linear is None:
           self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # x.register_hook(hook_function)
        # self.middleoutputs.append(x)

        x = x.view(-1,self._to_linear)

        x = F.sigmoid(self.fc1(x))

        # self.middleoutputs.append(x)
        # x.register_hook(hook_function)

        x = self.fc2(x) # 2
        # x.register_hook(hook_function)
        Output = F.sigmoid(x)

        return Output

# -------store parameters in each layer------#
def store_parameters(net):
    parameters = {}
    for name, parameter in net.named_parameters():
        # print(name, ":", parameter.size())
        parameters[name] = parameter.clone()
    return parameters

def hook_function(grad):
    gradient.append(grad.detach().numpy())

#-------train------#
def train_model(device, Train_X, Train_y):
    #-------train------#
    epochs = 2
    Batch_size = 1
    net = Net()

    parm = []
    output_histories = []
    loss_histories = []

    # -------send data to GPU, will not work if you are training on CPU------#
    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    loss_function = nn.MSELoss()


    for epoch in range(epochs):
        # for i in tqdm(range(0, len(Train_X), Batch_size)):
        for i in tqdm(range(1)):
            # print(i, i+Batch_size)
            batch_X = Train_X[i+1:i+1+Batch_size].view(-1,1,50,50)
            batch_y = Train_y[i+1:i+1+Batch_size]

            # -------employ training on CPU or GPU------#
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()

            # -------Record the Original Parameters------#
            parameters_org = store_parameters(net)
            parm.append(parameters_org)

            outputs = net(batch_X)

            output_histories.append(outputs)
            middleoutputs = net.middleoutputs
            convrealresults = net.convrealresults

            loss = loss_function(outputs,batch_y)
            loss_histories.append(loss)

            # -------Record Gradients after the first iteration------#
            loss.backward()
            optimizer.step()

            # -------Record Parameters after the first iteration------#
            # torch.save(net.state_dict(), PATH_AFT)
            parameters_in_this_iter = store_parameters(net)
            parm.append(parameters_in_this_iter)

            grad_histories = gradient
            print('ok')
    print(loss) #print out loss value

#-------test------#
def test_model(net,test_X,test_y):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1,1,50,50))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total +=1

    print("Accuracy", round(correct/total,3))

if __name__=='__main__':
    device = device_detect()
    Train_X, Train_y, test_X, test_y = data_preprocessing()
    train_model(device, Train_X,Train_y)