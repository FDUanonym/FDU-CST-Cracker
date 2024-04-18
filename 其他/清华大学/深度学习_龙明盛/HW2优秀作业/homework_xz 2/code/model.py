import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
import numpy as np
from performer_pytorch import Performer as PerformerModule


class LSTMCell(nn.Module):
    def __init__(self, ninput: int, nhid: int):
        super(LSTMCell, self).__init__()

        self.nhid = nhid
        self.w_x = nn.Parameter(torch.rand(4 * nhid, ninput) * 0.2 - 0.1)
        self.w_h = nn.Parameter(torch.rand(4 * nhid, nhid) * 0.2 - 0.1)
        self.bias = nn.Parameter(torch.rand(4 * nhid) * 0.2 - 0.1)
        self.w_o = nn.Parameter(torch.rand(nhid, nhid) * 0.2 - 0.1)

    def forward(self, x: torch.Tensor, h: torch.Tensor=None, c: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """
            x: [seq_len x batch_size x ninput]
            h: [batch_size x nhid]
            c: [batch_size x nhid]
        """
        if h is None:
            h = torch.zeros([x.size(1), self.nhid], dtype=torch.float, device=x.device)
        if c is None:
            c = torch.zeros([x.size(1), self.nhid], dtype=torch.float, device=x.device)
        
        pred = torch.empty([x.size(0), x.size(1), self.nhid], dtype=torch.float, device=x.device)
        
        for ts in range(x.size(0)):
            a = torch.einsum('ji,bi->bj', self.w_x, x[ts]) + torch.einsum('ji,bi->bj', self.w_h, h) + self.bias
            a_i, a_f, a_o, a_g = torch.split(a, a.size(-1) // 4, dim=-1)
            i, f, o, g = torch.sigmoid(a_i), torch.sigmoid(a_f), torch.sigmoid(a_o), torch.tanh(a_g)

            c = f * c + i * g
            h = o * torch.tanh(c)

            pred[ts] = torch.einsum('ji,bi->bj', self.w_o, h)

        return pred, h, c


class LSTM(nn.Module):
    def __init__(self, ninput: int, nhid: int, nlayers: int=1):
        super(LSTM, self).__init__()
        assert nlayers >= 1

        self.lstm_list = nn.ModuleList()

        self.lstm_list.append(LSTMCell(ninput, nhid))
        for _ in range(nlayers - 1):
            self.lstm_list.append(LSTMCell(nhid, nhid))

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]=(None, None)) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
            x: [seq_len x batch_size x ninput]
            h: [nlayers x batch_size x nhid]
            c: [nlayers x batch_size x nhid]
        """
        h, c = hidden

        h_out, c_out = [], []
        last_pred = x
        
        for i, net in enumerate(self.lstm_list):
            cur_h = h[i] if h is not None else None
            cur_c = c[i] if c is not None else None

            last_pred, cur_h, cur_c = net(last_pred)
            h_out.append(cur_h)
            c_out.append(cur_c)

        return last_pred, (torch.stack(h_out), torch.stack(c_out))


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_sql: int):
        super().__init__()

        # Postional Encoding according to Transformer
        encoding = torch.zeros(max_sql, 1, d_model)
        items = torch.arange(0, max_sql).float().unsqueeze(-1) * torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        encoding[:, 0, ::2] = torch.sin(items)
        encoding[:, 0, 1::2] = torch.cos(items)

        # Register in module to support .to(device)
        self.register_buffer('encoding', encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [seq_len x batch_size x d_model]
        """
        return x + self.encoding[:x.size(0)]


class Performer(nn.Module):
    def __init__(self, nvoc: int, ninput: int, nhid: int, nlayers: int, max_sql: int, nhead=2) -> None:
        super().__init__()

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput)

        # causal=True is equal to mask in Transformer
        self.encoder = PerformerModule(
            dim=ninput,
            depth=nlayers,
            heads=nhead,
            dim_head=nhid // nhead,
            causal=True
        )
        self.decoder = nn.Linear(ninput, nvoc)
        self.position_encoding = PositionEncoding(ninput, max_sql)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [seq_len x batch_size x d_model]
        """
        embed = self.embed(x)
        embed = self.position_encoding(embed)

        # embed: [batch_size x seq_len x d_model]
        embed = torch.transpose(embed, 0, 1)

        # y: [seq_len x batch_size x d_model]
        y = self.encoder(embed).transpose(0, 1)

        output = self.decoder(y)

        return output


class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers, model_type='gru'):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput) 
        
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        self.model_type = model_type
        if model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=ninput,
                hidden_size=nhid,
                num_layers=nlayers
            )
        elif model_type == 'lstm':
            self.rnn = LSTM(
                ninput=ninput,
                nhid=nhid,
                nlayers=nlayers
            )
        else:
            raise NotImplementedError(f"Unrecognized model type: {model_type}.")

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
        output, hidden = self.rnn(embeddings)

        ##################################################################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden



# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.attn_weights: torch.Tensor = None

    # Hook attn weights
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=True)
        self.attn_weights = x[1].detach()
        return self.dropout1(x[0])

    
class LMTransformer(nn.Module):
    def __init__(self, nvoc: int, ninput: int, nhid: int, nlayers: int, max_sql: int, nhead=2):
        super().__init__()

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayerWithWeights(
                d_model=ninput,
                nhead=nhead,
                dim_feedforward=nhid
            ),
            num_layers=nlayers
        )
        self.decoder = nn.Linear(ninput, nvoc)
        self.position_encoding = PositionEncoding(ninput, max_sql)

        # Attn mask
        attn_mask = torch.tril(torch.ones(max_sql, max_sql, dtype=torch.bool))
        attn_mask = attn_mask.float().masked_fill(~attn_mask, -np.inf).masked_fill(attn_mask, 0.0)

        # Register in buffer to support torch.device
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: torch.Tensor, need_weights: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embed = self.embed(x)
        y = self.encoder(self.position_encoding(embed), self.attn_mask[:x.size(0), :x.size(0 )])
        output = self.decoder(y)

        if need_weights:
            return output, self.encoder.layers[0].attn_weights
        else:
            return output

###########################################################################################
