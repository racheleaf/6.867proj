import torch
from torch import nn
from torch.autograd import Variable

"""
Note on inputs:
    LSTM expects inputs are 3D tensor sequence x minibatch x inputdata
    If you only have a single input you can run it with view(1,1,-1)
    Applying LSTM returns out, hidden
    out is entire history of hidden states
    hidden is most recent hidden state, for chainability

"""


def freeze_layer(layer):
    for parameter in layer.parameters():
        parameter.requires_grad = False


class TextLSTM(nn.Module):
    def __init__(self, input_size, lstm_size, output_size, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = lstm_size
        self.encoder = nn.Embedding(input_size, lstm_size)
        self.lstm = nn.LSTM(lstm_size, lstm_size, n_layers)
        self.decoder = nn.Linear(lstm_size, output_size)

    def forward(self, inp, hidden):
        batch_size = len(inp)
        enc = self.encoder(inp)
        output, hidden = self.lstm(enc.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
