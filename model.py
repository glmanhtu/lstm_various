"""
Define our RNN model as a new class inheriting from nn.Module. Our model will be a multilayer RNN followed by a linear
layer on the last output of RNN.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class SimpleRNN(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        dropout = 0 if num_layers == 1 else 0.05

        # Initialise the correct RNN layer depending on what we.
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers,
                              batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers,
                              batch_first=True)
        else:
            raise(ValueError('Incorrect choice of RNN supplied'))
        self.out = nn.Linear(hidden_size, 1)  # Linear layer is output of model

    def forward(self, x, x_length):
        # Define our forward pass, we take some input sequence and an initial hidden state.
        r_out, _ = self.rnn(x)

        final_y = self.out(r_out[:, -1, :])  # Return only the last output of RNN.

        return final_y


class LSTM(nn.Module):
    def __init__(self, CNN_embed_dim=1792, h_RNN_layers=1, h_RNN=512, h_FC_dim=128, drop_p=0.6, num_classes=1,
                 embedding=False):
        super(LSTM, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.RNN = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        self.embedding = embedding

    def forward(self, x_RNN, x_lengths):

        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_RNN[perm_idx], lengths_ordered, batch_first=True)
        self.RNN.flatten_parameters()
        packed_RNN_out, _ = self.RNN(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        if self.embedding:
            outs = []
            for i, length in enumerate(out_lengths):
                outs.append(RNN_out[i][:length])
            return outs
        idx = (lengths_ordered - 1).view(-1, 1).expand(len(lengths_ordered), RNN_out.size(2))
        time_dimension = 1  # Batch first = true, otherwise 2
        idx = idx.unsqueeze(time_dimension).type(torch.int64)
        if RNN_out.is_cuda:
            idx = idx.cuda(RNN_out.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = RNN_out.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))

        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        x = last_output[unperm_idx]

        # FC layers
        # I have commented this part to make sure it matched with the configuration in the original model above
        # For other more complex problems, consider to uncomment it

        # x = self.fc1(x)  # choose RNN_out at the last time step
        # x = F.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
