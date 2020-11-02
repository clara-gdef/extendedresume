import torch
import pytorch_lightning as pl
import ipdb
from allennlp.modules.elmo import batch_to_ids


class DecoderWithElmo(pl.LightningModule):
    def __init__(self, elmo, emb_dimension, hidden_size, num_layer, output_size):
        super().__init__()
        self.elmo = elmo
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = torch.nn.LSTM(emb_dimension * 2, hidden_size, num_layer,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, encoder_representation, token):
        ipdb.set_trace()
        enc_rep = encoder_representation
        character_ids = batch_to_ids(token)

        emb = self.elmo(character_ids.cuda())
        emb_tensor = emb["elmo_representations"][-1]

        inputs = torch.cat([enc_rep, emb_tensor], dim=2)
        out, hidden = self.lstm(inputs)
        results = self.lin_out(out)

        return results, hidden
