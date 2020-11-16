import torch
import pytorch_lightning as pl
import ipdb


class DecoderLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + 1, hidden_size, 1,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, encoder_representation, token, hidden_state):
        #enc_rep = encoder_representation.expand(token.shape[0], token.shape[1], encoder_representation.shape[-1]).transpose(1, 0)
        enc_rep = encoder_representation
        inputs = torch.cat([enc_rep.type(torch.float32), token.type(torch.float32).unsqueeze(-1).cuda()], dim=2)

        out, hidden = self.lstm(inputs, hidden_state)
        results = self.lin_out(out)

        return results, hidden
