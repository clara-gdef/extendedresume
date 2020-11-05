import torch
import pytorch_lightning as pl
import ipdb


class DecoderLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + 1, hidden_size, 1,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, encoder_representation, token):
        #enc_rep = encoder_representation.expand(token.shape[0], token.shape[1], encoder_representation.shape[-1]).transpose(1, 0)
        enc_rep = encoder_representation
        ipdb.set_trace()
        inputs = torch.cat([enc_rep.type(torch.float32).transpose(1, 0), token.type(torch.float32).unsqueeze(-1).transpose(1, 0).cuda()], dim=2)

        out, hidden = self.lstm(inputs)
        results = self.lin_out(out)

        return results.transpose(1, 0), hidden
