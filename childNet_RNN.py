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

class RNN_cell(nn.Module):
    def __init__(self, layers, hidden_unit,num_features, num_classes, layer_limit,actions):
        #hidden_layers：一个列表的激活函数，列表长度 = nodes数+1
        #DAGS：一个列表，存储有向图，每一个数字表示父节点，列表长度 = nodes数
        super(RNN_cell, self).__init__()
        self.hidd_unit = hidden_unit
        self.linear_layers = []#存储线性层的列表
        self.hidden_layers = []#存储激活函数的列表
        self.DAG = []
        self.actions = actions#controller的操作
        max_layers = 12
        if max_layers < layer_limit:
            raise Exception(
                'Maximum layers that ChildNet accepts is '.format(max_layers))

        self.linear_layers.append(
            nn.Linear(in_features=num_features, out_features=self.hidd_unit))#首先添加WX矩阵
        for i, layer in enumerate(layers):#按照给定的动作构建网络
            if layer == 'EOS':
                break
            elif isinstance(layer, int):#若是数字，则表明是
                self.linear_layers.append(
                    nn.Linear(in_features=self.hidd_unit,
                              out_features=self.hidd_unit))
                self.DAG.append(layer)
            else:
                self.hidden_layers.append(activation_functions[layer])
        #last layer must contain 2 out_features (2 classes)
        self.linear_layers.append(
            nn.Linear(in_features=self.hidd_unit, out_features=num_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.sum_weight = nn.Linear(in_features=len(actions), out_features=len(self.DAG))
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)#优化器
        self.dropout = nn.Dropout(p=0.2)#dropout层
        

    def get_father(self):
        return self.DAG
    
    def compute_val(self, x, h = None):
        x = x.to(torch.float32)#batch*embedding
        if h == None:
            h = torch.rand_like(self.linear_layers[0](x))#batch*hidden
        out_layers = {}
        output_used = [False
                       for u in range(len(self.linear_layers))]  #标记输出是否被使用
        out_layers[0] = self.linear_layers[0](x)#batch*hidden
        for i in range(1, len(self.linear_layers) - 1):
            prev = self.DAG[i - 1]
            output_used[prev] = True
            input_x = out_layers[prev]
            if i == 1:
                output = self.linear_layers[i](h.to(torch.float)) 
                output = output + input_x
                out_layers[i] = self.dropout(self.hidden_layers[i - 1](output))
            else:
                output = self.linear_layers[i](input_x)
                out_layers[i] = self.dropout(self.hidden_layers[i - 1](output))
        ans = []
        avg = 0
        
        layers = torch.from_numpy(np.array(self.actions)).to(torch.float)
        weight = self.sum_weight(layers)
        clean_weight = []
        for i in range(1, len(out_layers)):
            if output_used[i] == False:
                ans.append(out_layers[i])#used_layer*batch*hidden
                clean_weight.append(weight[i-1])
        ans = torch.stack(ans, axis=0).transpose(0,1).to(float)#batch*used_layer*hidden
        clean_weight = F.softmax(torch.stack(clean_weight, axis=0)).to(float)
        avg = torch.matmul(clean_weight, ans)
        h = torch.squeeze(avg, dim=1).to(torch.float)#batch*hidden
        hh = torch.mean(ans,dim=1).to(float)#batch*hidden
        return h
    
    def forward(self,word2vec,h,batch_input):
        for l in range(1,int(batch_input.shape[1])):#对RNN_cell执行循环
            x = batch_input[:,l].type(torch.long)
            h = self.compute_val(word2vec[x],h)
        avg = self.linear_layers[len(self.linear_layers) - 1](h.to(torch.float))#batch*class_num
        return avg#分类输出

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():  #定义
    def __init__(self, layer_limit,word2vec):
        self.criterion = nn.CrossEntropyLoss()

#         X_tr, y_tr, X_val, y_val, X_te, y_te, word2vec = create_dataset()
        self.word2vec = word2vec
        self.num_features = len(word2vec[0])
        self.num_classes = 2
        self.layer_limit = layer_limit

    def save_model(self, net, father):#保存模型，实现参数共享
        L = list(net.state_dict().keys())
        for i in range(len(L)):
            model_name = str(L[i])
            model_name = model_name.replace('.', '_')
            if model_name == 'sum_weight_weight' or model_name == 'sum_weight_bias':
                model_path = 'shared_model/RNN_NLP/' + model_name
            else:
                model_path = 'shared_model/RNN_NLP/' + model_name + '_from_' + str(
                    father[int(i / 2)])
            torch.save(net.state_dict()[L[i]], model_path)

    def load_model(self, net, father):#加载模型，实现参数共享
        L = list(net.state_dict().keys())
        for i in range(len(L)):
            model_name = str(L[i])
            model_name = model_name.replace('.', '_')
            if model_name == 'sum_weight_weight' or model_name == 'sum_weight_bias':
                model_path = 'shared_model/RNN_NLP/' + model_name
            else:
                model_path = 'shared_model/RNN_NLP/' + model_name + '_from_' + str(
                    father[int(i / 2)])
            if os.path.exists(model_path):
                model = torch.load(model_path)
                net.load_state_dict({L[i]: model}, strict=False)
        
    def compute_reward(self, layers, num_epochs,actions,train_loader,validation_loader):#计算reward返回给controller
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        patience = 10
        hidden_unit = 8
        net = RNN_cell(layers, hidden_unit, self.num_features, self.num_classes,
                       self.layer_limit,actions)
        #         print(net.DAG)
        net.train()
        max_val_acc = 0
        father = net.get_father()
        father = [-1] + father + [-2]
        weight_reset(net)
        self.load_model(net, father)
        patient_count = 0
        # training loop
        
        for e in range(num_epochs):
            print('child epoch {},computing loss'.format(e+1))
            for batch_idx, (x, y) in enumerate(train_loader):#子网络的训练
        # predict by running forward pass
                batch_input = x
                # predict by running forward pass
                batch_targets = y
                h = None
                tr_output = net(self.word2vec,h,batch_input)
                tr_output = F.softmax(tr_output,dim = 1)
                tr_loss = self.criterion(tr_output.float(), batch_targets.long())
                # zeroize accumulated gradients in parameters
                net.optimizer.zero_grad()
                # compute gradients given loss
                tr_loss.backward()
                # update the parameters given the computed gradients
                net.optimizer.step()
                train_losses.append(tr_loss.data.numpy())
            acc = 0
            for batch_idx, (x, y) in enumerate(validation_loader):#子网络的评价
                #AFTER TRAINING
                h = None
                # predict with validation input
                batch_input = x
                # predict by running forward pass
                batch_targets = y
                val_output = net(self.word2vec,h,batch_input)
                val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1)
                acc += val_output.eq(batch_targets.view_as(val_output)).sum().item()    # 记得加item()
            val_acc = acc/len(validation_loader.dataset)
            val_accuracies.append(val_acc)
            print('validation accurancy: {:6.2f}'.format(val_acc))
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
        self.save_model(net, father)
        return val_acc
        
    def test_accurancy(self, layers, actions,test_loader):#测试模块，用于测试最终的准确率
        # store loss and accuracy for information
        self.x_test = x_test
        self.y_test = y_test
        
        patience = 10
        hidden_unit = 128
        net = RNN_cell(layers, hidden_unit, self.num_features, self.num_classes,
                       self.layer_limit,actions)
        net.eval()
        max_val_acc = 0
        father = net.get_father()
        father = [-1] + father + [-2]
        self.load_model(net, father)

        # get test input and expected output as torch Variables and make sure type is correct
        acc = 0
        for batch_idx, (x, y) in enumerate(test_loader):
                #AFTER TRAINING
            h = None
            # predict with validation input
            batch_input = x
            # predict by running forward pass
            batch_targets = y
            test_output = net(self.word2vec,h,batch_input)
            test_output = torch.argmax(F.softmax(test_output, dim=-1), dim=-1)
            acc += test_output.eq(batch_targets.view_as(test_output)).sum().item()    # 记得加item()
        test_acc = acc/len(test_loader.dataset)

        return test_acc  #max_val_acc#**3 #-float(val_loss.detach().numpy())