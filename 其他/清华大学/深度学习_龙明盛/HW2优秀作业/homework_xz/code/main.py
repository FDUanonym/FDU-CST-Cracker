import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os
import os.path as osp
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import pandas as pd
import time


parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--model_type', type=str, default='gru',
                    help='Model type')
parser.add_argument('--epochs', type=int, default=40,
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
parser.add_argument('--input_size', type=int, default=64,
                    help='embedding size of words')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='hidden size of words')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used', default=2)
parser.add_argument('--log_file', type=str, default=None, help='log filename')

# feel free to add some other arguments
args = parser.parse_args()
logger.info(f'Model: {args.model_type}, Max Sql: {args.max_sql}')

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
logger.info('Building model...')

if args.model_type in ['gru', 'lstm']:
    net = model.RNN(
        nvoc=len(data_loader.vocabulary),
        ninput=args.input_size,
        nhid=args.hidden_size,
        nlayers=6,
        model_type=args.model_type
    ).to(device)
elif args.model_type in ['transformer']:
    net = model.LMTransformer(
        nvoc=len(data_loader.vocabulary),
        ninput=args.input_size,
        nhid=args.hidden_size,
        nlayers=6,
        max_sql=args.max_sql
    ).to(device)
elif args.model_type in ['performer']:
    net = model.Performer(
        nvoc=len(data_loader.vocabulary),
        ninput=args.input_size,
        nhid=args.hidden_size,
        nlayers=6,
        max_sql=args.max_sql
    ).to(device)
else:
    raise NotImplementedError(f"Unrecognized model type: {args.model_type}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=net.parameters())

curve_df = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': [],
    'time': []
}

################################################################################################



# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate():
    net.eval()
    data_loader.set_valid()

    with torch.no_grad():
        total_loss, total_step = 0, 0
        total_correct, total_words = 0, 0
        
        with tqdm(range(data_loader.valid_batch_num // args.max_sql), desc='Evaluation') as t:
            while True:
                t.update()
                data, target, end_flag = data_loader.get_batch()
                data = data.to(device)
                target = target.to(device)

                if args.model_type in ['gru', 'lstm']:
                    pred, hidden = net(data)
                else:
                    pred = net(data)

                # Reshape (pred [seq_len x batch x vocab] -> [seq_len x vocab x batch], target [-1] -> [seq_len x batch])
                pred = pred.transpose(1, 2)
                target = target.reshape(pred.size(0), -1)

                loss = torch.exp(torch.mean(criterion(pred, target), dim=0)).mean()

                total_loss += loss.item()
                total_step += 1

                total_correct += torch.sum(torch.argmax(pred, dim=1) == target).item()
                total_words += target.size(0) * target.size(1)

                if end_flag:
                    break

        net.train()
        data_loader.set_train()

        logger.info(f'Valid Loss: {total_loss / total_step}, Valid Acc: {total_correct / total_words}')

    return total_loss / total_step, total_correct / total_words
################################################################################################



# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################
def train():
    total_loss, total_step = 0, 0
    total_correct, total_words = 0, 0
    
    with tqdm(range(data_loader.train_batch_num // args.max_sql), desc='Training') as t:
        while True:
            t.update()
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)

            if args.model_type in ['gru', 'lstm']:
                pred, hidden = net(data)
            else:
                pred = net(data)

            # Reshape (pred [seq_len x batch x vocab] -> [seq_len x vocab x batch], target [-1] -> [seq_len x batch])
            pred = pred.transpose(1, 2)
            target = target.reshape(pred.size(0), -1)

            loss = torch.exp(torch.mean(criterion(pred, target), dim=0)).mean()

            total_loss += loss.item()
            total_step += 1
            t.set_description(f'Train Loss: {loss.item():.2f}')

            total_correct += torch.sum(torch.argmax(pred, dim=1) == target).item()
            total_words += target.size(0) * target.size(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if end_flag:
                break
        
    logger.info(f'Train Loss: {total_loss / total_step}, Train Acc: {total_correct / total_words}')
    return total_loss / total_step, total_correct / total_words

################################################################################################



# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
logger.info('Start training...')
data_loader.set_train()
net.train()

os.makedirs('logs', exist_ok=True)
log_file = args.log_file if args.log_file is not None else f'log_{args.model_type}_{args.max_sql}.csv'

best_valid_acc = 0.0

for epoch in range(1, args.epochs + 1):
    logger.info(f'--------------> Epoch ({epoch}/{args.epochs})')
    start_time = time.perf_counter()
    train_loss, train_acc = train()
    end_time = time.perf_counter()
    valid_loss, valid_acc = evaluate()

    # Save log
    curve_df['epoch'].append(epoch)
    curve_df['train_loss'].append(train_loss)
    curve_df['train_acc'].append(train_acc)
    curve_df['valid_loss'].append(valid_loss)
    curve_df['valid_acc'].append(valid_acc)
    curve_df['time'].append(end_time - start_time)
    pd.DataFrame(curve_df).to_csv(os.path.join('logs', log_file), index=False)

    # Save model
    if valid_acc > best_valid_acc:
        logger.info(f'Best Acc! Saving model...')
        os.makedirs('best_model', exist_ok=True)
        torch.save(net.state_dict(), os.path.join('best_model', f'{args.model_type}_{args.max_sql}.pth'))
    
################################################################################################
