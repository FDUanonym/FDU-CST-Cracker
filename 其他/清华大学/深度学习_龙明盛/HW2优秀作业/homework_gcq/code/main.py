import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data
import model
import os
import os.path as osp
from time import perf_counter


parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default = 50,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default = 32, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default = 128, metavar='N',
                    help='eval batch size')
parser.add_argument("--input_shape", type = int, default = 512, help = "input state shape")
parser.add_argument("--hidden_shape", type = int, default = 256, help = "hidden state size")
parser.add_argument("--layer_num", type = int, default = 1, help = "layer number")

parser.add_argument("--learning_rate", type = float, default = 0.001, help = "learning rate")
# you can increase the seqence length to see how well the model works when capturing long-term dependencies
parser.add_argument('--max_sql', type=int, default=35, 
                    help='sequence length')
parser.add_argument('--seed', type=int, default = 42,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default = "0", help='GPU device id used')
parser.add_argument("--mode", type = str, default = "RNN", help = "model selection")

# feel free to add some other arguments
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/wikitext2", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar                                                           #
# Build model, optimizer and so on                                                             #
################################################################################################

nvoc = len(data_loader.vocabulary)
if args.mode == "RNN":
    mymodel = model.RNN(nvoc, args.input_shape, args.hidden_shape, args.layer_num)
elif args.mode == "LSTM":
    mymodel = model.LSTM(nvoc, args.input_shape, args.hidden_shape)
elif args.mode == "Transformer":
    mymodel = model.LMTransformer(nvoc, args.input_shape, args.hidden_shape, nelayers = args.layer_num)
mymodel = mymodel.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr = args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)

################################################################################################



# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate():
    mymodel.eval()
    data_loader.set_valid()
    end_flag = False
    val_loss = 0.0
    acc_top1 = 0.0
    acc_top10 = 0.0
    count = 0

    with torch.no_grad():
        while not end_flag:
            input, target, end_flag = data_loader.get_batch()
            input = input.to(device)
            target = target.to(device)
            output, _ = mymodel(input)
            
            output = output.reshape((-1, nvoc))
            loss = criterion(output.reshape((-1, nvoc)), target)
            _, predict1 = torch.max(output, 1)
            _, predict10 = torch.topk(output, 10, dim = 1)

            val_loss += loss.item() * target.shape[0]
            acc_top1 += torch.sum(predict1 == target.data).item()
            acc_top10 += torch.sum(torch.eq(predict10, target.reshape(-1, 1)).any(dim = 1)).item()
            count += target.shape[0]

    val_loss /= count
    acc_top1 /= count
    acc_top10 /= count
    return val_loss, acc_top1, acc_top10

################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################
def train():
    mymodel.train()
    data_loader.set_train()
    end_flag = False
    train_loss = 0.0
    acc_top1 = 0.0
    acc_top10 = 0.0
    count = 0

    while not end_flag:
        input, target, end_flag = data_loader.get_batch()
        optimizer.zero_grad()
        input = input.to(device)
        target = target.to(device)
        output, _ = mymodel(input)
        
        output = output.reshape((-1, nvoc))
        loss = criterion(output, target)
        _, predict1 = torch.max(output, 1)
        _, predict10 = torch.topk(output, 10, dim = 1)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.shape[0]
        acc_top1 += torch.sum(predict1 == target.data).item()
        acc_top10 += torch.sum(torch.eq(predict10, target.reshape(-1, 1)).any(dim = 1)).item()
        count += target.shape[0]

    train_loss /= count
    acc_top1 /= count
    acc_top10 /= count
    return train_loss, acc_top1, acc_top10

################################################################################################




# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
if __name__ == "__main__": 
    train_loss_list = []
    train_acc1_list = []
    train_acc10_list = []
    val_loss_list = []
    val_acc1_list = []
    val_acc10_list = []

    tic = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc1, train_acc10 = train()
        val_loss, val_acc1, val_acc10 = evaluate()

        train_loss_list.append(train_loss)
        train_acc1_list.append(train_acc1)
        train_acc10_list.append(train_acc10)
        val_loss_list.append(val_loss)
        val_acc1_list.append(val_acc1)
        val_acc10_list.append(val_acc10)

        scheduler.step()
        toc = perf_counter()
        print(f"""Epoch {epoch}/{args.epochs} 
        Training: loss: {train_loss:.5f}, top 1 acc: {train_acc1: .5f}, top 10 acc: {train_acc10: .5f}
        Validation: loss: {val_loss:.5f}, top 1 acc: {val_acc1: .5f}, top 10 acc: {val_acc10: .5f}
        Time elapsed: {toc - tic:.2f}s""")

    plt.plot(np.arange(1, args.epochs + 1), train_loss_list, label = "Training loss")
    plt.plot(np.arange(1, args.epochs + 1), val_loss_list, label = "Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(args.mode + "_curves_loss.png")

    plt.figure()
    plt.plot(np.arange(1, args.epochs + 1), train_acc1_list, label = "Training top 1 acc.")
    plt.plot(np.arange(1, args.epochs + 1), val_acc1_list, label = "Validation top 1 acc.")
    plt.xlabel("epochs")
    plt.ylabel("top 1 accuracy")
    plt.legend()
    plt.savefig(args.mode + "_top1_acc.png")

    plt.figure()
    plt.plot(np.arange(1, args.epochs + 1), train_acc10_list, label = "Training top 10 acc.")
    plt.plot(np.arange(1, args.epochs + 1), val_acc10_list, label = "Validation top 10 acc.")
    plt.xlabel("epochs")
    plt.ylabel("top 10 accuracy")
    plt.legend()
    plt.savefig(args.mode + "_top10_acc.png")

    result_df = pd.DataFrame({"train_loss": train_loss_list, "train_top1_acc": train_acc1_list, "train_top10_acc": train_acc10_list, "val_loss": val_loss_list, "val_top1_acc": val_acc1_list, "val_top10_acc": val_acc10_list}, index = None)
    result_df.to_csv(args.mode + "_results.csv")

################################################################################################