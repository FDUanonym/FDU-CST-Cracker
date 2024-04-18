import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

import data
import model
import os
import os.path as osp
import matplotlib.pyplot as plt
from logger import Logger
from genericpath import exists

parser = argparse.ArgumentParser(description='PyTorch Language Model')
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
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=1, help='GPU device id used')
parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
parser.add_argument('--log_dir', type=str, default='./logs/', help='log save path')
parser.add_argument('--model', type=str, default='Transformer', help='model')
parser.add_argument('--visualize_attn', default=False, help='visualize attention weight of transformer')
parser.add_argument('--adversarial_training', default=True, help='transformer with adversarial training')
parser.add_argument('--model_comment', type=str, default='model_Transformer', help='model comment')

# feel free to add some other arguments
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = args.cuda

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

logging = True
if args.model == 'GRU' or args.model == 'LSTM':
    ninput = 200
    nhid = 200
    nlayers = 2
    lr = 0.001
    epsilon = 0.1 # For adversarial training
    model_LM = model.RNN(len(data_loader.vocabulary), ninput, nhid, nlayers, args.model, adversarial_training=args.adversarial_training, epsilon=epsilon)
elif args.model == 'Transformer':
    ninput = 200
    nhid = 800
    nlayers = 2
    nhead = 8
    lr = 0.001
    epsilon = 0.005 # For adversarial training
    visualize_attn = args.visualize_attn
    model_LM = model.LMTransformer(len(data_loader.vocabulary), ninput, nhid, nlayers, nhead, adversarial_training=args.adversarial_training, epsilon=epsilon)
model_LM = model_LM.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_LM.parameters(), lr=lr)

def transformer_visualize_attn():
    # line = 'The cat stuck out its tongue and <unk> its owner . . The cat stuck out its tongue and <unk> its owner . The cat stuck out its tongue and <unk> its owner .'
    # line = line.split() + ['<eos>']
    # data = torch.LongTensor([data_loader.word_id[l] for l in line]).unsqueeze(1)
    model_LM.train(False)
    data_loader.set_valid()
    for i in range(4):
        data, _, _ = data_loader.get_batch()
    for l in range(nlayers):
        model_LM.transformer_encoder_layers[l].self_attn.visualize = True
        model_LM.transformer_encoder_layers[l].self_attn.query_text = [data_loader.vocabulary[data[i, 0]] for i in range(args.max_sql)]
        model_LM.transformer_encoder_layers[l].self_attn.key_text = [data_loader.vocabulary[data[i, 0]] for i in range(args.max_sql)]
        model_LM.transformer_encoder_layers[l].self_attn.layer = l + 1
    _, _ = model_LM(data.to(device))
    for l in range(nlayers):
        model_LM.transformer_encoder_layers[l].self_attn.visualize = False
        model_LM.transformer_encoder_layers[l].self_attn.query_text = None
        model_LM.transformer_encoder_layers[l].self_attn.key_text = None
    # plt.show()
################################################################################################



# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate():
    model_LM.train(False)
    total_loss = 0.0

    data_loader.set_valid()
    end_flag = False
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        outputs, _ = model_LM(data)
        outputs = outputs.view(-1, len(data_loader.vocabulary))
        loss = criterion(outputs, target)

        total_loss += loss.item() * outputs.size(0)
    
    epoch_loss = total_loss / data_loader.valid.size(0) / data_loader.valid.size(1)
    epoch_perplexity = math.exp(epoch_loss)

    return epoch_loss, epoch_perplexity
################################################################################################




# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################
def train():
    model_LM.train(True)
    total_loss = 0.0

    data_loader.set_train()
    end_flag = False
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        if args.adversarial_training == True:
            outputs, _ = model_LM(data, target)
        else:
            outputs, _ = model_LM(data)
        outputs = outputs.view(-1, len(data_loader.vocabulary))
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * outputs.size(0)
        
    epoch_loss = total_loss / data_loader.train.size(0) / data_loader.train.size(1)
    epoch_perplexity = math.exp(epoch_loss)

    return epoch_loss, epoch_perplexity
################################################################################################




# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
best_perplexity = float('inf')
model_comment = args.model_comment
save_path = osp.join(args.save_dir, model_comment)
log_path = osp.join(args.log_dir, model_comment)

if not exists(save_path):
    os.makedirs(save_path)
if logging:
    if not exists(log_path):
        os.makedirs(log_path)
    logger = Logger(log_path)

for epoch in range(1, args.epochs+1):
    print('*' * 100)
    print('model: {}'.format(model_comment))
    print('epoch: {:d}/{:d}'.format(epoch, args.epochs))
    start_time = time.time()
    train_loss, train_perplexity = train()
    train_time = time.time() - start_time
    start_time = time.time()
    valid_loss, valid_perplexity = evaluate()
    valid_time = time.time() - start_time
    print("training: {:.4f}, {:.4f}, {:.4f}s/epoch".format(train_loss, train_perplexity, train_time))
    print("validation: {:.4f}, {:.4f}, {:.4f}s/epoch".format(valid_loss, valid_perplexity, valid_time))
    if valid_perplexity < best_perplexity:
        best_perplexity = valid_perplexity
        best_model = model_LM
        torch.save(best_model, osp.join(save_path, 'best_model.pt'))
    if logging:
        logger.log_value('train_loss', train_loss)
        logger.log_value('train_perplexity', train_perplexity)
        logger.log_value('valid_loss', valid_loss)
        logger.log_value('valid_perplexity', valid_perplexity)
        logger.step()

if (args.model == 'Transformer') and visualize_attn:
    transformer_visualize_attn()
################################################################################################
