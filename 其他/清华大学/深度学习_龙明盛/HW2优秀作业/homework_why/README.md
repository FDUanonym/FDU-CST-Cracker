# 运行说明

问题解答与实验报告均在report.pdf中

main.py的运行结果将会保存在args.dir路径下

## Part One

1. 将`main.py`中model设置为"RNN", choice设置为"GRU"，可调节词向量维度"ninput", 层数"nlayers"，隐藏层维度"nhid"

2. 将`main.py`中model设置为"RNN", choice设置为"LSTM"，可调节词向量维度"ninput", 层数"nlayers"，隐藏层维度"nhid"

## Part Two

1. **Multihead-Attention**部分可以直接运行`mha.py`，使用pytorch版本为: 1.13.0+cu117
2. **Transformer for Language Modeling**部分将model设置为"transformer"，可以调节词向量维度"ninput"，隐藏层维度"nhid"，encoder层数"en_layers"，decoder层数"de_layers"，注意力头数"nhead"，feedforward维度"dim_ff"
3. **Attention Visualization:**在jupyter中运行attnvis.ipynb即可，其中`model.load_state_dict(torch.load('../transformer/best_model.pt'))`需要更改为训练好transformer所在的路径

## Part Three

自定义的Gaussian多头注意力模块在`mMHA.py`中实现

在`main.py`中需要将"guassian"设置为True，其余参数同上所述。