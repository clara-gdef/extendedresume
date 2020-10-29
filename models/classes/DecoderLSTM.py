import torch
import pytorch_lightning as pl
from torch.nn import functional as F


class DecoderLSTM(pl.LightningModule):
    def __init__(self, embeddings, hidden_size):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(embeddings.size(0), 100, padding_idx=0)
        self.embedding_layer.load_state_dict({'weight': embeddings[:, :100]})

        self.lstm = torch.nn.LSTM(embeddings.shape[-1] + (hidden_size * 2), hidden_size * 2, 1,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size * 2, embeddings.shape[0])

    def forward(self, encoder_representation, hidden_state, token):
        enc_rep = encoder_representation

        emb = self.embedding_layer(token)
        inputs = torch.cat([enc_rep, emb], dim=2)
        out, hidden = self.lstm(inputs, hidden_state)
        results = self.lin_out(out)

        return results, hidden
