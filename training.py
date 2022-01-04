import torch
from childNet_RNN import ChildNet
from utils import fill_tensor, indexes_to_actions
from torch.autograd import Variable
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences 
import os
# 转化为 DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
def training(policy, batch_size, total_actions,worc2vec,train_loader,validation_loader,verbose = False, num_episodes = 500):
    ''' Optimization/training loop of the policy net. Returns the trained policy. '''
    
    # training settings
    decay = 0.9
    training = True
    val_freq = num_episodes/5
    cn = ChildNet(policy.layer_limit,worc2vec)
    nb_epochs = 5
    
    # train policy network
    training_rewards, val_rewards, losses = [], [], []
    baseline = torch.zeros(15, dtype=torch.float)
    best_r = 0
    print('start training')
    for i in range(num_episodes):
        print('Epoch {}'.format(i+1))
        rollout, batch_r, batch_a_probs = [], [], []
        #forward pass
        with torch.no_grad():
            actions, activation_prob = policy(training)#获取LSTM的输出，建立children network
            #修改，再获得一个DAG以及其对应的概率
        batch_hid_units, batch_index_eos = indexes_to_actions(actions, batch_size, total_actions)
        #compute individually the rewards
        for j in range(batch_size):#debug
            # policy gradient update 
            print('batch_size {}, computing reward'.format(j+1))
            r = cn.compute_reward(batch_hid_units[j], nb_epochs,actions[j],train_loader,validation_loader)**3#计算子网络的reward,当成自己的action
#             print('mean validation accurancy: {:6.2f}'.format(r))
            if batch_hid_units[j]==['EOS']:
                r -= -1
            if best_r < r:
                best_r = r
                policy.save_model
            a_probs = activation_prob[j, :batch_index_eos[j] + 1]
            
            batch_r += [r]#每一个reward
            batch_a_probs += [a_probs.view(1, -1)]#每一个batch中，每一个动作被选中的概率

        #rearrange the action probabilities
#         print(batch_a_probs)
        a_probs = []
        for b in range(batch_size):
            a_probs.append(fill_tensor(batch_a_probs[b], policy.n_outputs, ones=True))
        a_probs = torch.stack(a_probs,0)
        #convert to pytorch tensors --> use get_variable from utils if training in GPU
        batch_a_probs = Variable(a_probs, requires_grad=True)
        batch_r = Variable(torch.tensor(batch_r), requires_grad=True)#获得的reward，需要继续用到
        loss = policy.loss(batch_a_probs, batch_r, torch.mean(baseline))
        policy.optimizer.zero_grad()  
        loss.backward()
        policy.optimizer.step()

        # actualize baseline
        baseline = torch.cat((baseline[1:]*decay, torch.tensor([torch.mean(batch_r)*(1-decay)], dtype=torch.float)))
        
        # bookkeeping
        training_rewards.append(torch.mean(batch_r).detach().numpy())
        losses.append(loss.item())        
        # print training
        if verbose :#and (i+1) % val_freq == 0:
            print('{:4d}. mean training reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, torch.mean(batch_r).detach().numpy(), loss.item()))
        
    print('done training')  
    return policy

def create_dataset(p_val=0.1, p_test=0.2):#创建数据集，将它放到更外层有利于减小开销
    import numpy as np
    import sklearn.datasets
    MAX_WORDS = 10000
    MAX_LEN = 100
    emb_size = 128
    child_batch = 5
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS,maxlen=MAX_LEN)
    
    x = x_train.tolist() + x_test.tolist()
    y = y_train.tolist() + y_test.tolist()
    train_end = int(len(x) * (1 - p_val - p_test))
    val_end = int(len(x) * (1 - p_test)) 
   #--------数据清洗-----------
    
    # define train, validation, and test sets
    x_train = x[:train_end]
    x_validation = x[train_end:val_end]
    x_test = x[val_end:]

    # and labels
    y_train = y[:train_end]
    y_validation = y[train_end:val_end]
    y_test = y[val_end:]
    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    #-------准备word2vec--------
    dic = set()
    for line in range(0, len(x)):
        for word in x[line]:
            dic.add(word)
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=x,
                     vector_size=emb_size,
                     window=5,
                     min_count=1,
                     workers=4)
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    word2vec = [model.wv[word] for word in dic]
    while len(word2vec) < MAX_WORDS:
        word2vec.append([0.0] * emb_size)
    word2vec = torch.from_numpy(np.array(word2vec))
    #----------------------准备dataloader-------------------------
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_validation = pad_sequences(x_validation, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))#打包数据
    validation_data = TensorDataset(torch.LongTensor(x_validation), torch.LongTensor(y_validation))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=child_batch)
    validation_sampler = SequentialSampler(validation_data)
    validation_loader = DataLoader(validation_data, sampler=validation_sampler, batch_size=child_batch)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=child_batch)
    return train_loader,validation_loader, test_loader, word2vec


def testing(policy, total_actions,worc2vec,x_test,y_test):
    ''' Optimization/training loop of the policy net. Returns the trained policy. '''
    
    # training settings
    training = False
    cn = ChildNet(policy.layer_limit,worc2vec)
    
    # train policy network
    training_rewards, val_rewards, losses = [], [], []
    baseline = torch.zeros(15, dtype=torch.float)
    
    print('start testing')
    #forward pass
    with torch.no_grad():
        actions, activation_prob = policy(training)#获取LSTM的输出，建立children network
        #修改，再获得一个DAG以及其对应的概率
    batch_hid_units, batch_index_eos = indexes_to_actions(actions, 1, total_actions)
    r = cn.test_accurancy(batch_hid_units[0], actions[0],test_loader)
    # print training
    print('mean test accurancy: {:6.2f}'.format(r))
    print('done testing')  
    
def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
            