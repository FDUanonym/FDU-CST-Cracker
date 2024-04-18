# _*_ coding:utf-8 _*_
from genericpath import exists
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import data
import model
import os
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
# from bertviz import head_view, model_view

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default=80,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
# you can increase the seqence length to see how well the model works when capturing long-term dependencies
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--dirs', type=str, default='../test', help='save directions')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--milestones', type=int, default=[100], help='decay epochs')
parser.add_argument('--model', type=str, default='transformer', help='RNN or transformer')
parser.add_argument('--nvoc', type=int, default=33278, help='size of dictionary embeddings')
parser.add_argument('--en_layers', type=int, default=4, help='layer of encoder')
parser.add_argument('--de_layers', type=int, default=6, help='layer of decoder')
parser.add_argument('--nhead', type=int, default=8, help='head in transformer')
parser.add_argument('--dim_ff', type=int, default=2048, help='feedforward dimensions in transformer')
parser.add_argument('--ninput', type=int, default=400, help='the size of each embedding vector')
parser.add_argument('--choice', type=str, default='LSTM', help='model choice, GRU or LSTM')
parser.add_argument('--nhid', type=int, default=400, help='hidden dimensions')
parser.add_argument('--nlayers', type=int, default=2, help='layer nums in RNN')
parser.add_argument('--guassian', type=bool, default=False, help='gaussian transformer or not')
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
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/wikitext2", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar                                                           #
# Build model, optimizer and so on                                                             #
################################################################################################

criterion = nn.CrossEntropyLoss()
if args.model == 'RNN':
    model = model.RNN(args.nvoc, args.ninput, args.nhid, args.nlayers, args.choice)
    model = model.to(device)
if args.model == 'transformer':
    model = model.LMTransformer(args.nvoc, args.ninput, args.nhid, args.en_layers, args.de_layers, args.dim_ff, args.nhead, args.guassian)
    model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=0.01)
if not exists(args.dirs):
    os.makedirs(args.dirs)


################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate(model, data_loader, criterion):
    data_loader.set_valid()
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    end_flag = False
    while not end_flag:
        data, label, end_flag = data_loader.get_batch()
        data, label = data.to(device), label.to(device)
        if args.model == 'RNN':
            output, hidden = model(data)
            output = output.view(output.size(0) * output.size(1), -1)
        if args.model == 'transformer':
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(data.shape[0]).to(device)
            output = model(data, tgt_mask)
            output = output.view(output.size(0) * output.size(1), -1)
        loss = criterion(output, label)
        predictions = torch.argmax(output, 1)
        total_loss += loss.item() * label.size(0)
        total_correct += torch.sum(predictions == label.data)
    valid_loss = total_loss / data_loader.valid.numel()
    valid_acc = total_correct.double() / data_loader.valid.numel()
    return valid_loss, valid_acc


################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################

def train(args, data_loader, model, optimizer, criterion):
    ## training in one epoch
    data_loader.set_train()
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    end_flag = False
    # temp = torch.Tensor(range(args.max_sql)).unsqueeze(0).repeat(args.max_sql, 1)
    while not end_flag:
        data, label, end_flag = data_loader.get_batch()
        data, label = data.to(device), label.to(device)
        if args.model == 'RNN':
            output, hidden = model(data)
            output = output.view(output.size(0) * output.size(1), -1)
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(data.shape[0]).to(device)
            output = model(data, tgt_mask)
            output = output.view(output.size(0) * output.size(1), -1)
        loss = criterion(output, label)
        # print(loss)
        predictions = torch.argmax(output, dim=1)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        total_loss += loss.item() * label.size(0)
        total_correct += torch.sum(predictions == label.data)
    train_loss = total_loss / data_loader.train.numel()
    train_acc = total_correct.double() / data_loader.train.numel()
    return train_loss, train_acc


################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
train_loss = torch.zeros([args.epochs])
train_perplexity = torch.zeros([args.epochs])
train_acc = torch.zeros([args.epochs])
valid_loss = torch.zeros([args.epochs])
valid_perplexity = torch.zeros([args.epochs])
valid_acc = torch.zeros([args.epochs])
best_perplexity = 1e5
for epoch in range(args.epochs):
    print('epoch:{:d}/{:d}'.format(epoch + 1, args.epochs))
    print('*' * 100)
    st = time.time()
    train_loss[epoch], train_acc[epoch] = train(args, data_loader, model, optimizer, criterion)
    train_perplexity[epoch] = torch.exp(train_loss[epoch])
    print("training: {:.4f}, {:.4f}".format(train_loss[epoch], train_perplexity[epoch]))
    valid_loss[epoch], valid_acc[epoch] = evaluate(model, data_loader, criterion)
    et = time.time()
    valid_perplexity[epoch] = torch.exp(valid_loss[epoch])
    print("validation: {:.4f}, {:.4f}".format(valid_loss[epoch], valid_perplexity[epoch]))
    print("running time in one epoch={:.4f}s".format(et-st))
    if valid_perplexity[epoch].item() < best_perplexity:
        best_perplexity = valid_perplexity[epoch].item()
        best_model = model
        torch.save(best_model.state_dict(), os.path.join(args.dirs, 'best_model.pt'))
    scheduler.step(valid_perplexity[epoch].item())
torch.save(train_loss, os.path.join(args.dirs, 'train_loss.pt'))
torch.save(train_acc, os.path.join(args.dirs, 'train_acc.pt'))
torch.save(train_perplexity, os.path.join(args.dirs, 'train_perplexity.pt'))
torch.save(valid_loss, os.path.join(args.dirs, 'valid_loss.pt'))
torch.save(valid_acc, os.path.join(args.dirs, 'valid_acc.pt'))
torch.save(valid_perplexity, os.path.join(args.dirs, 'valid_perplexity.pt'))
epoch_range = torch.Tensor(range(args.epochs))
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epoch_range, train_loss, color='b', label="train")
plt.plot(epoch_range, valid_loss, color='r', label="test")
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.plot(epoch_range, train_perplexity, color='b', label="train")
plt.plot(epoch_range, valid_perplexity, color='r', label="test")
plt.legend()
plt.xlabel('epoch')
plt.ylabel('perplexity')
plt.savefig(args.dirs + 'curves.png')
################################################################################################