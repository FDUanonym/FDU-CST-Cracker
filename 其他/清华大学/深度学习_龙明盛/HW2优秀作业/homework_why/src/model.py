import torch
import torch.nn as nn
from mMHA import GuassianMHA


class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers, choice='GRU'):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput)

        # nvoc: size of dictionary embeddings
        # ninput: the size of each embedding vector
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        # self.rnn = None
        class mLSTMCell(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(mLSTMCell, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.W_x = torch.nn.Parameter(torch.Tensor(input_size, hidden_size, 4))
                self.W_h = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size, 4))
                self.bias = torch.nn.Parameter(torch.Tensor(hidden_size, 4))
                nn.init.kaiming_uniform_(self.W_x)
                nn.init.kaiming_uniform_(self.W_h)
                nn.init.kaiming_uniform_(self.bias)

            def forward(self, input):
                seq_len, batch_size = input.size(0), input.size(1)
                hidden_prev = torch.zeros([batch_size, self.hidden_size]).to(input.device)
                cell_prev = torch.zeros([batch_size, self.hidden_size]).to(input.device)
                hiddens = torch.zeros([seq_len, batch_size, self.hidden_size]).to(input.device)
                for t in range(seq_len):
                    it = torch.sigmoid(torch.matmul(input[t, :, :], self.W_x[:, :, 0]) + torch.matmul(hidden_prev,
                                                                                                      self.W_h[:, :,
                                                                                                      0]) + self.bias[:,
                                                                                                            0])
                    ft = torch.sigmoid(torch.matmul(input[t, :, :], self.W_x[:, :, 1]) + torch.matmul(hidden_prev,
                                                                                                      self.W_h[:, :,
                                                                                                      1]) + self.bias[:,
                                                                                                            1])
                    ot = torch.sigmoid(torch.matmul(input[t, :, :], self.W_x[:, :, 2]) + torch.matmul(hidden_prev,
                                                                                                      self.W_h[:, :,
                                                                                                      2]) + self.bias[:,
                                                                                                            2])
                    gt = torch.tanh(torch.matmul(input[t, :, :], self.W_x[:, :, 3]) + torch.matmul(hidden_prev,
                                                                                                   self.W_h[:, :,
                                                                                                   3]) + self.bias[:,
                                                                                                         3])
                    cell_new = ft * cell_prev + it * gt
                    hidden = ot * torch.tanh(cell_new)
                    hiddens[t, :, :] = hidden
                    hidden_prev, cell_prev = hidden, cell_new
                return hiddens, hidden

        self.rnn = torch.nn.ModuleList()
        for l in range(nlayers):
            if l == 0:
                rnn_layer = torch.nn.GRU(ninput, nhid, 1, dropout=0.5) if choice == 'GRU' else mLSTMCell(ninput, nhid)
            else:
                rnn_layer = torch.nn.GRU(nhid, nhid, 1, dropout=0.5) if choice == 'GRU' else mLSTMCell(nhid, nhid)
            self.rnn.append(rnn_layer)
        # each layer has the same hidden size nhid
        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
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
        hidden = []
        output = embeddings
        for l, rnn in enumerate(self.rnn):
            current_input = output
            output, new_h = rnn(current_input)
            hidden.append(new_h)
        # hidden = torch.Tensor(hidden)
        ##################################################################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
class LMTransformer(nn.Module):
    def __init__(self, nvoc, ninput, nhid, en_layers, de_layers, dim_ff, nhead, guassian):
        super(LMTransformer, self).__init__()
        self.embed = nn.Embedding(nvoc, ninput)
        self.linear = nn.Linear(nhid, nvoc)
        self.en_layers = en_layers
        self.de_layers = de_layers
        self.nhead = nhead
        self.transformer = nn.Transformer(d_model=nhid, num_encoder_layers=en_layers, num_decoder_layers=de_layers, dim_feedforward=dim_ff, nhead=nhead, dropout=0.5)
        if guassian:
            for ld in range(de_layers):
                self.transformer.decoder.layers[ld].multihead_attn = GuassianMHA(nhid, nhead, dropout=0.5)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_ff, dropout=0.5)
        # self.encoder = nn.ModuleList([self.encoder_layer for i in range(en_layers)])
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_ff, dropout=0.5)
        # self.decoder_layer = nn.ModuleList([self.decoder_layer for i in range(de_layers)])
        # self.transformer = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=de_layers)
        self.drop = nn.Dropout(0.5)

    def forward(self, input, tgt_mask):
        embeddings = self.drop(self.embed(input))
        output = self.transformer(embeddings, embeddings, tgt_mask=tgt_mask)
        output = self.drop(output)
        decoded = self.linear(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def visEncoderAttention(self, x, layer):
        assert layer < self.en_layers, "encoder layer exceed!"
        # assert head < self.nhead, "head number exceed!"
        output = self.embed(x)
        for l in range(layer):
            output = self.transformer.encoder.layers[l](output)
        vis_layer = self.transformer.encoder.layers[layer]
        # output = vis_layer.norm1(output)
        _, attention_weight = vis_layer.self_attn(output, output, output, need_weights=True)
        return attention_weight

    def visDecoderAttention(self, x, layer, tgt_mask):
        assert layer < self.de_layers, "encoder layer exceed!"
        # assert head < self.nhead, "head number exceed!"
        output = self.embed(x)
        memory = self.transformer.encoder(output)
        for l in range(layer):
            output = self.transformer.decoder.layers[l](output, memory, tgt_mask=tgt_mask)
        vis_layer = self.transformer.decoder.layers[layer]
        # output = vis_layer.norm1(output)
        output = vis_layer.norm1(output+vis_layer.self_attn(output, output, output, attn_mask=tgt_mask)[0])
        _, attention_weight = vis_layer.multihead_attn(output, memory, memory, need_weights=True)
        return attention_weight
###########################################################################################