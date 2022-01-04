import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class PolicyNet(nn.Module):#定义controller
    """Policy network, i.e., RNN controller that generates the different childNet architectures."""

    def __init__(self, batch_size, n_outputs, layer_limit):
        super(PolicyNet, self).__init__()
        
        # parameters
        self.layer_limit = layer_limit
        self.gamma = 1.0
        self.n_hidden = 24
        self.n_outputs = n_outputs
        self.learning_rate = 1e-2
        self.batch_size = batch_size
        # Neural Network
        self.lstm = nn.LSTMCell(self.n_outputs, self.n_hidden)
        self.linear_classifer = nn.Linear(self.n_hidden, self.n_outputs)
        
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def one_hot(self, t, num_classes):
        '''One hot encoder of an action/hyperparameter that will be used as input for the next RNN iteration. '''
        out = np.zeros((t.shape[0], num_classes))
        for row, col in enumerate(t):
            out[row, col] = 1
        return out.astype('float32')

    def sample(self, output, training):
        '''Stochasticity of the policy, picks a random action based on the probabilities computed by the last softmax layer. '''
        if training:#训练状态，根据概率随机选取
            random_array = np.random.rand(self.batch_size).reshape(self.batch_size,1)
            return (np.cumsum(output.detach().numpy(), axis=1) > random_array).argmax(axis=1) # sample action
        else: #测试状态，选择概率最大的
            return (output.detach().numpy()).argmax(axis=1)
        
    def save_model(self):
        L = list(self.state_dict().keys())
        for i in range(len(L)):
            model_name = str(L[i])
            model_name = model_name.replace('.', '_')
            model_path = 'shared_model/RNN_NLP/controller/' + model_name
            torch.save(net.state_dict()[L[i]], model_path)
            
    def forward(self, training):
        ''' Forward pass. Generates different childNet architectures (nb of architectures = batch_size). '''
        outputs = []
        prob = []
        actions = np.zeros((self.batch_size, self.n_outputs))
        action = not None #initialize action to don't break the while condition 
        i = 0
        counter_nb_layers = 0
        father = torch.zeros(self.batch_size, self.n_outputs, dtype=torch.float)
        h_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float)
        c_t = torch.zeros(self.batch_size, self.n_hidden, dtype=torch.float)
        action = torch.zeros(self.batch_size, self.n_outputs, dtype=torch.float)
        
        while counter_nb_layers<self.layer_limit and i <self.n_outputs:#根据LSTM的概率输出，获取网络的结构
            if counter_nb_layers > 0:
                h_t, c_t = self.lstm(action, (h_t, c_t))
                DAG_output = F.softmax(self.linear_classifer(h_t))
                DAG_output = F.softmax(DAG_output[...,self.n_outputs-self.layer_limit:self.n_outputs-self.layer_limit+counter_nb_layers])
                father = self.sample(DAG_output, training)#根据output按照概率选择下一步动作
                
                outputs += [DAG_output]
                prob.append(DAG_output[np.arange(self.batch_size),father])
                father = father + self.n_outputs-self.layer_limit
                actions[:, i] = father
                father = torch.tensor(self.one_hot(father, self.n_outputs))        
                i += 1
            #----------------------------------------------
            h_t, c_t = self.lstm(father, (h_t, c_t))
            output = F.softmax(self.linear_classifer(h_t))
            output = F.softmax(output[...,:self.n_outputs-self.layer_limit-1])
            action = self.sample(output, training)#根据output按照概率选择下一步动作
            outputs += [output]
            prob.append(output[np.arange(self.batch_size),action])
            action = action+1
            actions[:, i] = action
            action = torch.tensor(self.one_hot(action, self.n_outputs))            
            i += 1
            counter_nb_layers += 1
        prob = torch.stack(prob, 1)#压缩维度
#         print(prob)
        return actions,prob

    def loss(self, action_probabilities, returns, baseline):#梯度下降
        ''' Policy loss. More details see the article uploaded in https://github.com/RualPerez/AutoML '''
        #T is the number of hyperparameters 
        sum_over_T = torch.sum(torch.log(action_probabilities.view(self.batch_size, -1)), axis=1)#分别取log再求和
        subs_baseline = torch.add(returns,-baseline)#减去基准线后
        return torch.mean(torch.mul(sum_over_T, subs_baseline)) - torch.sum(torch.mul (torch.tensor(0.01) * action_probabilities, torch.log(action_probabilities.view(self.batch_size, -1))))
