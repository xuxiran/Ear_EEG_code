import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math


# --------------------------CNN-baseline-----------------------------
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,20), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=4)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)

        return fc2_out









# -------------------STAnet---------------------------

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)


        self.W_q = nn.Linear(query_size, num_hiddens * 10, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens * 10, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)

        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)


        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)

        output = self.attention(queries, keys, values)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)

        output_concat = transpose_output(output, self.num_heads)


        return output_concat


class EEG_STANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_STANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        R_c = self.conv1(E)
        R_s = self.pooling1(self.elu(R_c))
        M_s = self.linear2(self.elu(self.linear1(R_s)))


        Ep = M_s * E


        Ep = Ep.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        Eps = Eps.squeeze(dim=1)
        E_t = self.attention(Eps, Eps, Eps)


        E_t = E_t.reshape(E_t.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out


class EEG_SANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_SANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        R_c = self.conv1(E)
        R_s = self.pooling1(self.elu(R_c))
        M_s = self.linear2(self.elu(self.linear1(R_s)))


        Ep = M_s * E


        Ep = Ep.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        # Eps = Eps.squeeze(dim=1)
        # E_t = self.attention(Eps, Eps, Eps)


        E_t = Eps.reshape(Eps.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out


class EEG_TANet(nn.Module):

    def __init__(self, channel_num=16):
        super(EEG_TANet, self).__init__()

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(128, 1), stride=(1, 1))
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.linear1 = nn.Linear(20, 8)
        self.dropout = 0.5

        self.elu = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.linear2 = nn.Linear(8, 20)

        # conv block
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1))

        self.tanh = nn.Sequential(
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
            nn.Dropout(p=self.dropout)
        )


        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)


        self.fc1 = nn.Linear(160, 4)

    def forward(self, E):
        E = E.unsqueeze(dim=1)


        # R_c = self.conv1(E)
        # R_s = self.pooling1(self.elu(R_c))
        # M_s = self.linear2(self.elu(self.linear1(R_s)))
        #

        # Ep = M_s * E
        #

        Ep = E.permute(0, 3, 2, 1)
        Epc = self.conv2(Ep)
        Epc = Epc.permute(0, 3, 2, 1)
        Eps = self.pooling2(self.tanh(Epc))


        Eps = Eps.squeeze(dim=1)
        E_t = self.attention(Eps, Eps, Eps)


        E_t = E_t.reshape(E_t.shape[0], -1)
        final_out = self.fc1(E_t)

        return final_out