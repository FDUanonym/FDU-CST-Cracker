# Code for Homework 2
## 运行GRU:
```sh
python3 main.py --model_type gru
```

## 运行LSTM:
```sh
python3 main.py --model_type lstm
```

## 运行Transformer:
```sh
python3 main.py --model_type transformer
```

## 运行Performer:
- 请先安装`pip3 install performer-pytorch`。
- 如果需要使用CUDA，确保`nvcc`命令可以正常使用并编译cuda文件。运行`pip3 install pytorch-fast-transformers`。
- 运行如下命令：
```sh
python3 main.py --model_type performer
```
