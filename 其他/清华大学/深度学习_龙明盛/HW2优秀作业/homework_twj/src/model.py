import torch
import torch.nn as nn
from mha import MultiHeadAttention

class LSTM(nn.Module):
    def __init__(self, ninput, nhid, nlayers):
        super(LSTM, self).__init__()
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc = [nn.Linear(nhid + ninput, 4 * nhid)]
        for l in range(1, nlayers):
            self.fc.append(nn.Linear(nhid + nhid, 4 * nhid))
        self.fc = nn.ModuleList(self.fc)

    def forward(self, input):
        seq_len, batch_size = input.size(0), input.size(1)
        output = torch.zeros(seq_len, batch_size, self.nhid, dtype=input.dtype, device=input.device)
        hidden = torch.zeros(self.nlayers, batch_size, self.nhid, dtype=input.dtype, device=input.device)
        cell = torch.zeros(self.nlayers, batch_size, self.nhid, dtype=input.dtype, device=input.device)
        for i in range(seq_len):
            for l in range(self.nlayers):
                if l == 0:
                    activation = self.fc[0](torch.cat((input[i, :, :], hidden[0, :, :]), dim=1))
                else:
                    activation = self.fc[l](torch.cat((hidden[l-1, :, :], hidden[l, :, :]), dim=1))
                input_gate = self.sigmoid(activation[:, :self.nhid])
                forget_gate = self.sigmoid(activation[:, self.nhid:2*self.nhid])
                output_gate = self.sigmoid(activation[:, 2*self.nhid:3*self.nhid])
                block_gate = self.tanh(activation[:, 3*self.nhid:])
                cell[l, :, :] = forget_gate * cell[l, :, :].clone() + input_gate * block_gate
                hidden[l, :, :] = output_gate * self.tanh(cell[l, :, :])
            output[i, :, :] = hidden[-1, :, :]
        
        return output, hidden

class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers, type='GRU', adversarial_training=False, epsilon=0.005):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput) 
        
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        if type == 'GRU':
            self.rnn = nn.GRU(ninput, nhid, nlayers) # For GRU
        elif type == 'LSTM':
            self.rnn = LSTM(ninput, nhid, nlayers) # For LSTM
        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

        # For adversarial training
        self.adversarial_training = adversarial_training
        self.epsilon = epsilon

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    # feel free to change the forward arguments if necessary
    def forward(self, input, target=None):
        embeddings = self.drop(self.embed(input))

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        output, hidden = self.rnn(embeddings)
        ##################################################################################

        output = self.drop(output)
        if self.adversarial_training and self.training:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            _output = output.view(output.size(0)*output.size(1), output.size(2))

            weight_noise = torch.zeros_like(self.decoder.weight).to(self.decoder.weight.device)
            neg_h = -_output / torch.sqrt(torch.sum(_output**2, 1, keepdim=True) + 1e-8)
            n_output = torch.sqrt(torch.sum(_output**2, 1, keepdim=True) + 1e-8)
            n_w = torch.sqrt(torch.sum(self.embed(target)**2, 1, keepdim=True) + 1e-8)
            cos_theta = (torch.sum(_output * self.embed(target), 1, keepdim=True)) / n_output / n_w
            indicator = torch.gt(cos_theta, 0e-1).view(-1, 1).type(torch.FloatTensor).to(cos_theta.device)
            sigma = self.epsilon * n_w * indicator
            weight_noise[target.view(-1)] = sigma.detach() * neg_h.detach()
            noise_outputs = (_output * weight_noise[target]).sum(1)
            decoded[torch.arange(target.size(0)).long().to(target.device), target] += noise_outputs
        else:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden



# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, ninput, nhead, nhid, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(ninput, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(ninput, nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid, ninput)

        self.norm1 = nn.LayerNorm(ninput)
        self.norm2 = nn.LayerNorm(ninput)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, input_mask=None):
        input_T = input.transpose(0, 1) # (N, T, E) -> (T, N, E)
        input2 = self.self_attn(input_T, input_T, input_T, attn_mask=input_mask)
        input = input + self.dropout1(input2.transpose(0, 1))
        input = self.norm1(input)
        input2 = self.linear2(self.dropout(self.relu(self.linear1(input))))
        input = input + self.dropout2(input2)
        output = self.norm2(input)
        return output

class LMTransformer(nn.Module):
    def __init__(self, nvoc, ninput, nhid, nlayers, nhead, dropout=0.1, max_sql=100, adversarial_training=False, epsilon=0.005):
        super(LMTransformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embed = nn.Embedding(nvoc, ninput) 
        self.transformer_encoder_layers = nn.ModuleList([TransformerEncoderLayer(ninput, nhead, nhid, dropout) for i in range(nlayers)])
        self.decoder = nn.Linear(ninput, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        # Positional encoding generation
        pe = torch.zeros(max_sql, ninput)
        term = torch.arange(0, max_sql, dtype=torch.float).unsqueeze(1) / torch.pow(10000, torch.arange(0, ninput, 2, dtype=torch.float) / ninput)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(term), torch.cos(term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        # Mask generation
        mask = torch.tril(torch.ones(max_sql, max_sql))
        self.register_buffer('mask', mask)

        # For adversarial training
        self.adversarial_training = adversarial_training
        self.epsilon = epsilon

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, target=None):
        embeddings = self.drop(self.embed(input))
        input = embeddings + self.pe[:embeddings.size(0), :, :]

        mask = self.mask[:embeddings.size(0), :embeddings.size(0)]
        for layer in self.transformer_encoder_layers:
            input = layer(input, input_mask=mask)

        output = self.drop(input)

        if self.adversarial_training and self.training:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            _output = output.view(output.size(0)*output.size(1), output.size(2))

            weight_noise = torch.zeros_like(self.decoder.weight).to(self.decoder.weight.device)
            neg_h = -_output / torch.sqrt(torch.sum(_output**2, 1, keepdim=True) + 1e-8)
            n_output = torch.sqrt(torch.sum(_output**2, 1, keepdim=True) + 1e-8)
            n_w = torch.sqrt(torch.sum(self.embed(target)**2, 1, keepdim=True) + 1e-8)
            cos_theta = (torch.sum(_output * self.embed(target), 1, keepdim=True)) / n_output / n_w
            indicator = torch.gt(cos_theta, 0e-1).view(-1, 1).type(torch.FloatTensor).to(cos_theta.device)
            sigma = self.epsilon * n_w * indicator
            weight_noise[target.view(-1)] = sigma.detach() * neg_h.detach()
            noise_outputs = (_output * weight_noise[target]).sum(1)
            decoded[torch.arange(target.size(0)).long().to(target.device), target] += noise_outputs
        else:
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), None

###########################################################################################
