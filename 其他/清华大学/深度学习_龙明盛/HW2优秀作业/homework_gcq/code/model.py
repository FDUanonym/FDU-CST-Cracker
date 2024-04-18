import torch
import torch.nn as nn
import math
from math import sqrt

class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput) 
        
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        self.rnn = torch.nn.GRU(ninput, nhid, nlayers, batch_first = False)

        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    # feel free to change the forward arguments if necessary
    def forward(self, input):
        embeddings = self.drop(self.embed(input))
        # print(embeddings.shape)

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        output, hidden = self.rnn(embeddings)

        ##################################################################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

class LSTM(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput) 
        
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        self.W = nn.Parameter(torch.Tensor(ninput, nhid * 4))
        self.U = nn.Parameter(torch.Tensor(nhid, nhid * 4))
        self.b = nn.Parameter(torch.Tensor(nhid * 4))

        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1

        self.W.data.uniform_(-1 / sqrt(self.nhid), 1 / sqrt(self.nhid))
        self.U.data.uniform_(-1 / sqrt(self.nhid), 1 / sqrt(self.nhid))

        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    # feel free to change the forward arguments if necessary
    def forward(self, input):
        embeddings = self.drop(self.embed(input))

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        
        sql_len, batch_size, _ = embeddings.shape
        output = torch.zeros((sql_len, batch_size, self.nhid)).to(input.device)
        h_t, c_t = (torch.zeros(batch_size, self.nhid).to(input.device),
                    torch.zeros(batch_size, self.nhid).to(input.device))
        
        for t in range(sql_len):
            x_t = embeddings[t, :, :]
            gates = x_t @ self.W + h_t @ self.U + self.b
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :self.nhid]), # input
                torch.sigmoid(gates[:, self.nhid:2 * self.nhid]), # forget
                torch.tanh(gates[:, 2 * self.nhid:3 * self.nhid]), # gate
                torch.sigmoid(gates[:, 3 * self.nhid:4 * self.nhid]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            output[t, :, :] = h_t

        ##################################################################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (h_t, c_t)


# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
class LMTransformer(nn.Module):
    
    def __init__(self, nvoc, ninput, nhid, nhead = 8, nelayers = 2, ndlayers = 2, dropout = 0.2):
        super(LMTransformer, self).__init__()
        self.model_type = "Transformer"
        self.ninput = ninput

        self.embed = nn.Embedding(nvoc, ninput) 
        self.pos_encoder = PositionalEncoding(ninput, dropout)
        encode_layers = nn.TransformerEncoderLayer(ninput, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layers, nelayers)
        self.decoder = nn.Linear(ninput, nvoc)
        
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        sql_len, _ = input.shape
        # https://torchtutorialstaging.z5.web.core.windows.net/beginner/transformer_tutorial.html 
        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
        mask = torch.triu(torch.ones(sql_len, sql_len) * float("-inf"), diagonal = 1).to(input.device)

        embeddings = self.embed(input) * math.sqrt(self.ninput)
        embeddings = self.pos_encoder(embeddings)
        output = self.transformer_encoder(embeddings, mask)
        decoded = self.decoder(output)
        return decoded, None

class PositionalEncoding(nn.Module):

    def __init__(self, ninput, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ninput, 2) * (-math.log(10000.0) / ninput))
        pe = torch.zeros(max_len, 1, ninput)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input):
        input = input + self.pe[:input.size(0)]
        return self.dropout(input)

###########################################################################################

# GPT implementation from scratch
# https://jaketae.github.io/study/gpt/

class GPT(nn.Module):

    def __init__(self, nvoc, ninput, nhid, nlayers):
        super().__init__()
        self.embed = nn.Embedding(nvoc, ninput)
        self.pos_encoder = PositionalEncoding(ninput)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential(
            *[GPTBlock(ninput, nhid) for _ in range(nlayers)]
        )
        self.ln = nn.LayerNorm(ninput)
        self.decoder = nn.Linear(ninput, nvoc)

    def forward(self, input):
        embeddings = self.embed(input)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.blocks.to(input.device)(embeddings)
        embeddings = self.ln(embeddings)
        decoded = self.decoder(embeddings)
        return decoded, None

class GPTBlock(nn.Module):

    def __init__(self, ninput, nhid):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(ninput)
        self.ln2 = nn.LayerNorm(ninput)
        self.attn = nn.MultiheadAttention(ninput, 8)
        self.ff = nn.Sequential(
            nn.Linear(ninput, nhid),
            nn.GELU(),
            nn.Linear(nhid, ninput),
            nn.Dropout(0.1)
        )
        
    def forward(self, input):
        input = self.ln1(input)
        sql_len = input.size(0)
        mask = torch.triu(torch.ones(sql_len, sql_len) * float("-inf"), diagonal = 1).to(input.device)
        output, _ = self.attn(input, input, input, attn_mask = mask)
        output = input + output
        output = output + self.ff(self.ln2(output))
        return output
