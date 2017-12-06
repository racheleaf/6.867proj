from torch import nn

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
        super()
        self.encoder = nn.Embedding(input_size, lstm_size)
        self.lstm = nn.LSTM(input_size, lstm_size, n_layers)
        self.decoder = nn.Linear(lstm_size, output_size)

    def forward(self, inp, hidden):
        enc = self.encoder(inp.view(
