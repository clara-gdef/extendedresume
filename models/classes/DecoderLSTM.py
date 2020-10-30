import torch
import pytorch_lightning as pl
from torch.nn import functional as F


class DecoderLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + 1, hidden_size, 1,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, encoder_representation, token):
        inputs = torch.cat([encoder_representation, token.type(torch.FloatTensor).unsqueeze(-1).cuda()], dim=2)

        out, hidden = self.lstm(inputs)
        results = self.lin_out(out)

        return results, hidden
