from policy import PolicyNet
from training import * 
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch 


if __name__ == "__main__":
        
    # input parameters
    parser = argparse.ArgumentParser(description='Documentation in the following link: https://github.com/RualPerez/AutoML', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch', help='Batch size of the policy (int)', nargs='?', const=1, type=int, default=15)
    parser.add_argument('--max_layer', help='Maximum nb layers of the childNet (int)', nargs='?', const=1, type=int, default=6)
    parser.add_argument('--possible_act_functions', default=['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU'], nargs='*', 
                        type=int, help='Possible activation funcs of the childnet (list of str)')
    parser.add_argument('--verbose', help='Verbose while training the controller/policy (bool)', nargs='?', const=1, 
                        type=bool, default=False)
    parser.add_argument('--num_episodes', help='Nb of episodes the policy net is trained (int)', nargs='?', const=1, 
                        type=int, default=500)
    args = parser.parse_args()
    
    PATH = "shared_model/RNN_NLP/"
    del_file(PATH)#删除文件夹中存在的文件，以防输入参数被改变时报错
    
    DAGspace = [ i for i in range(args.max_layer)]#有向图合法的父节点
    args.possible_act_functions = ['EOS'] + args.possible_act_functions + DAGspace

    total_actions = args.possible_act_functions#定义动作空间：合法的父节点+激活函数+EOS
    n_outputs = len(args.possible_act_functions) #of the PolicyNet
    # setup policy network
    policy = PolicyNet(args.batch, n_outputs, args.max_layer)
    train_loader,validation_loader, test_loader,worc2vec = create_dataset()
    # train
    policy = training(policy, args.batch, total_actions, worc2vec,train_loader,validation_loader,args.verbose, args.num_episodes)
    
    testing(policy, total_actions,worc2vec, test_loader)
    
    # save model
    torch.save(policy.state_dict(), 'policy.pt')