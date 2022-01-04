import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
activation_functions = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Linear': nn.Identity()
}

def create_dataset(p_val=0.1, p_test=0.2):
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 1000

    X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    
    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))
    
    # define train, validation, and test sets
    X_tr = X[:train_end]
    X_val = X[train_end:val_end]
    X_te = X[val_end:]

    # and labels
    y_tr = y[:train_end]
    y_val = y[train_end:val_end]
    y_te = y[val_end:]

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val#？test去哪了

class Net(nn.Module):

    def __init__(self, layers, num_features, num_classes, layer_limit): 
        #hidden_layers：一个列表的激活函数，列表长度 = nodes数+1
        #DAGS：一个列表，存储有向图，每一个数字表示父节点，列表长度 = nodes数
        super(Net, self).__init__()
        self.hidd_unit = 2
        self.linear_layers = []
        self.hidden_layers = []
        self.DAG = []
        max_layers = 12
        if max_layers < layer_limit:
            raise Exception('Maximum layers that ChildNet accepts is '.format(max_layers))

        self.linear_layers.append(nn.Linear(in_features=num_features, out_features=self.hidd_unit))
        for i,layer in enumerate(layers):
            if layer == 'EOS':
                break
            elif isinstance(layer, int):
                self.linear_layers.append(nn.Linear(in_features=self.hidd_unit, out_features=self.hidd_unit))
                self.DAG.append(layer)
            else:
                self.hidden_layers.append(activation_functions[layer])
        #last layer must contain 2 out_features (2 classes)
        self.linear_layers.append(nn.Linear(in_features=self.hidd_unit, out_features=num_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        
    def get_father(self):
        return self.DAG
    
    def forward(self, x):
        out_layers = {}
        out_layers[0] = self.linear_layers[0](x)
        print(out_layers[0].shape)
        output_used = [False for u in range(len(self.linear_layers))]#标记输出是否被使用
        for i in range(1,len(self.linear_layers)-1):
            prev = self.DAG[i-1]
            output_used[prev] = True
            input_x = out_layers[prev]
            output = self.linear_layers[i](input_x)
            out_layers[i] = self.hidden_layers[i-1](output)
        ans = []
        avg = 0
        for i in range(1,len(out_layers)):
            if output_used[i] == False:
                ans.append(out_layers[i])
        ans = torch.stack(ans,axis=0)
        avg = torch.mean(ans,axis = 0)
        avg = self.linear_layers[len(self.linear_layers)-1](avg)
        return avg
    
def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(ts.long(), torch.max(ys, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())
    
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():#大体上不用修改

    def __init__(self, layer_limit):
        self.criterion = nn.CrossEntropyLoss()

        X_tr, y_tr, X_val, y_val = create_dataset()
        self.X_tr = X_tr.astype('float32')
        self.y_tr = y_tr.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        
        self.num_features = X_tr.shape[-1]
        self.num_classes = 2
        self.layer_limit = layer_limit
    
    def save_model(self,net,father):
        L = list(net.state_dict().keys())
        for i in range(len(L)):
            model_name = str(L[i])
            model_name = model_name.replace('.','_')
            model_path = 'shared_model/linear/'+ model_name+'_from_' + str(father[int(i/2)])
            torch.save(net.state_dict()[L[i]],model_path)
     
    def load_model(self,net,father):
#         os.path.exists()
#         net.load_state_dict({'fea.0.0.weight':model['features.0.weight']}, strict=False)
        L = list(net.state_dict().keys())
        for i in range(len(L)):
            model_name = str(L[i])
            model_name = model_name.replace('.','_')
            model_path = 'shared_model/linear/'+ model_name+'_from_' + str(father[int(i/2)])
            if os.path.exists(model_path):
                model = torch.load(model_path)
                net.load_state_dict({L[i]:model}, strict=False)
#         fuck
    
    def compute_reward(self, layers, num_epochs):
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        patience = 10
        
        net = Net(layers,self.num_features, self.num_classes, self.layer_limit)
#         print(net.DAG)
        max_val_acc = 0
        father = net.get_father()
        father = [-1]+father+[-2]
        self.load_model(net,father)
        
        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr))
        tr_targets = Variable(torch.from_numpy(self.y_tr))

        # get validation input and expected output as torch Variables and make sure type is correct
        val_input = Variable(torch.from_numpy(self.X_val))
        val_targets = Variable(torch.from_numpy(self.y_val))

        patient_count = 0
        # training loop
        for e in range(num_epochs):

            # predict by running forward pass
            tr_output = net(tr_input)
            # compute cross entropy loss
            #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            # zeroize accumulated gradients in parameters
            net.optimizer.zero_grad()
            # compute gradients given loss
            tr_loss.backward()
            #print(net.l_1.weight.grad)
            # update the parameters given the computed gradients
            net.optimizer.step()
            
            train_losses.append(tr_loss.data.numpy())

            #AFTER TRAINING

            # predict with validation input
            val_output = net(val_input)
            val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1)
            
            # compute loss and accuracy
            #val_loss = self.criterion(val_output.float(), val_targets.long())
            val_acc = torch.mean(torch.eq(val_output, val_targets.type(torch.LongTensor)).type(torch.FloatTensor))
            
            #accuracy(val_output, val_targets)
            val_acc = float(val_acc.numpy())
            val_accuracies.append(val_acc)
            
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
        self.save_model(net,father)
            
        return val_acc#max_val_acc#**3 #-float(val_loss.detach().numpy()) 
