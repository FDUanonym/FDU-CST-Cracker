import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # Construct a new MultiHeadAttention layer.

        # Inputs:
        #  - embed_dim: Dimension of the token embedding
        #  - num_heads: Number of attention heads
        #  - dropout: Dropout probability
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head
        self.visualize = False
        self.query_text = None
        self.key_text = None
        self.layer = 0

    def forward(self, query, key, value, attn_mask=None):
        # Calculate the masked attention output for the provided data, computing
        # all attention heads in parallel.

        # In the shape definitions below, N is the batch size, S is the source
        # sequence length, T is the target sequence length, and E is the embedding
        # dimension.

        # Inputs:
        # - query: Input data to be used as the query, of shape (N, S, E)
        # - key: Input data to be used as the key, of shape (N, T, E)
        # - value: Input data to be used as the value, of shape (N, T, E)
        # - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
        #   i in the source should not influence token j in the target.

        # Returns:
        # - output: Tensor of shape (N, S, E) giving the weighted combination of
        #   data in value according to the attention weights calculated using key
        #   and query.
        N, S, E = query.shape
        N, T, E = value.shape

        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))

        # WRITE CODE HERE within two '#' bar                                       #
        # Implement multiheaded attention using the equations given in             #
        # homework document.                                                       #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        H, EH = self.n_head, self.head_dim
        # Convert dimension (0, 1, 2, 3) -> (0, 2, 1, 3) for convenience
        key_T = self.key(key).view((N, T, H, EH)).transpose(1, 2).transpose(2, 3)
        query = self.query(query).view((N, S, H, EH)).transpose(1, 2)
        value = self.value(value).view((N, T, H, EH)).transpose(1, 2)
        attn_weights = torch.matmul(query, key_T) / np.sqrt(EH)
        if attn_mask is not None:
            # Set masked value alpha_ij to -inf so that exp(alpha_ij) == 0
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=3)

        if self.visualize:
            # For attention weights visualization
            NS = 6
            plt.figure(figsize=(20, 9*(H//2)))
            for h in range(H):
                attn_weights_numpy = attn_weights[0, h, :, :].cpu().detach().numpy()
                plt.subplot(H//2, 2, h+1)
                plt.imshow(attn_weights_numpy)
                plt.xticks(np.arange(T), self.key_text, rotation=90)
                plt.xlabel('Key')
                plt.yticks(np.arange(S), self.query_text)
                plt.ylabel('Query')
                plt.colorbar()
                plt.title('Attention weights of layer {}, head {}'.format(self.layer, h + 1))
            plt.savefig('{}_attention_weight_layer{}.svg'.format(self.n_head, self.layer))
            for h in range(H):
                attn_weights_numpy = attn_weights[0, h, :, :].cpu().detach().numpy()
                plt.figure(figsize=(20, 8))
                plt.suptitle('Attention weights visualization of layer {}, head {}'.format(self.layer, h + 1))
                for s in range(NS):
                    plt.subplot(1, NS, s+1)
                    plt.axis('off')
                    for ss in range(S):
                        alpha = 1 if ss == s else 0.1
                        plt.text(0.48, -(2*ss - S), self.query_text[ss], alpha = alpha, horizontalalignment='right')
                    for t in range(T):
                        alpha = attn_weights_numpy[s, t]
                        plt.plot([0.5, 1.5], [-(2*s - S), -(2*t - T)], 'k', alpha=alpha)
                        plt.text(1.52, -(2*t - T), self.key_text[t], alpha=alpha, horizontalalignment='left')
                    plt.xlim((0, 2))
                plt.tight_layout()
                plt.savefig('{}_attention_weight_visualization_layer{}_head{}_start.svg'.format(self.n_head, self.layer, h + 1))

                plt.figure(figsize=(20, 8))
                plt.suptitle('Attention weights visualization of layer {}, head {}'.format(self.layer, h + 1))
                for s in range(S-NS, S):
                    plt.subplot(1, NS, (s-(S-NS))+1)
                    plt.axis('off')
                    for ss in range(S):
                        alpha = 1 if ss == s else 0.1
                        plt.text(0.48, -(2*ss - S), self.query_text[ss], alpha = alpha, horizontalalignment='right')
                    for t in range(T):
                        alpha = attn_weights_numpy[s, t]
                        plt.plot([0.5, 1.5], [-(2*s - S), -(2*t - T)], 'k', alpha=alpha)
                        plt.text(1.52, -(2*t - T), self.key_text[t], alpha=alpha, horizontalalignment='left')
                    plt.xlim((0, 2))
                plt.tight_layout()
                plt.savefig('{}_attention_weight_visualization_layer{}_head{}_end.svg'.format(self.n_head, self.layer, h + 1))

        attn_weights = self.attn_drop(attn_weights)
        # Convert dimension back: (0, 2, 1, 3) -> (0, 1, 2, 3)
        output = torch.matmul(attn_weights, value).transpose(1, 2).reshape((N, S, E))
        output = self.proj(output)
        ############################################################################
        return output


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':
    torch.manual_seed(1234)

    # Choose dimensions such that they are all unique for easier debugging:
    # Specifically, the following values correspond to N=1, H=2, T=3, E//H=4, and E=8.
    batch_size = 1
    sequence_length = 3
    embed_dim = 8
    attn = MultiHeadAttention(embed_dim, num_heads=2)

    # Self-attention.
    data = torch.randn(batch_size, sequence_length, embed_dim)
    self_attn_output = attn(query=data, key=data, value=data)

    # Masked self-attention.
    mask = torch.randn(sequence_length, sequence_length) < 0.5
    masked_self_attn_output = attn(query=data, key=data, value=data, attn_mask=mask)

    # Attention using two inputs.
    other_data = torch.randn(batch_size, sequence_length, embed_dim)
    attn_output = attn(query=data, key=other_data, value=other_data)

    expected_self_attn_output = np.asarray([[
      [-0.30784,-0.08971,0.57260,0.19825,0.08959,0.28221,-0.05153,-0.23268]
      ,[-0.35230,0.10586,0.42247,0.09276,0.13765,0.11636,-0.09490,0.01749]
      ,[-0.30555,-0.23511,0.78333,0.37899,0.26324,0.13141,-0.00239,-0.20351]]])

    expected_masked_self_attn_output = np.asarray([[
      [-0.34701,0.07956,0.40117,-0.00986,0.07048,0.26159,-0.13170,-0.06824]
      ,[-0.26902,-0.53240,0.73553,0.24340,0.12136,0.56356,0.01649,-0.51939]
      ,[-0.23963,-0.00882,0.75761,0.27159,0.16656,0.10638,-0.09657,-0.11547]]])

    expected_attn_output = np.asarray([[
      [-0.483236,0.206833,0.392467,0.031948,0.155175,0.179157,-0.118605,0.049207]
      ,[-0.214869,0.205259,0.261078,0.154042,-0.045083,0.147627,0.077088,-0.050551]
      ,[-0.393120,0.158911,0.252667,0.132215,0.083187,0.254064,0.000776,-0.117547]]])

    print('self_attn_output error: ', rel_error(expected_self_attn_output, self_attn_output.detach().numpy()))
    print('masked_self_attn_output error: ', rel_error(expected_masked_self_attn_output, masked_self_attn_output.detach().numpy()))
    print('attn_output error: ', rel_error(expected_attn_output, attn_output.detach().numpy()))
