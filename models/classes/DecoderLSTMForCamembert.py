import torch
import pytorch_lightning as pl
import ipdb


class DecoderLSTMForCamembert(pl.LightningModule):
    def __init__(self, input_size, hidden_size, emb_dim, out_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + emb_dim, hidden_size, num_layers=1,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, encoder_representation, emb_token, hidden_state):
        enc_rep = encoder_representation
        inputs = torch.cat([enc_rep.type(torch.float32), emb_token.type(torch.float32).cuda()], dim=2)

        out, hidden = self.lstm(inputs, hidden_state)
        results = self.lin_out(out)

        return results, hidden
