# ENAS_NLP
 中山大学2021秋季强化学习期末项目

配置环境：```pip3 install -r requirements.txt```

训练模型：python3 main.py --num_episodes [训练轮数] --batch [batch大小] --max_layer[子网络最大层数] --possible_act_functions[子网络可用的激活函数] --verbose[是否显示信息] --num_episodes[训练的轮数]

均有默认值，可直接键入python3 main.py 

相关文件结构：

shared_model：存储模型

Sentiment_Analysis_Based_On_ENAS.pdf：实验报告

childNet_RNN.py：RNN相关的子网络代码

main.py：主函数

policy.py：ENAS搜索使用到的LSTM网络的代码

training.py：训练，获取数据以及测试代码

utils：辅助函数

word2vec.model：NLP用到的模型
