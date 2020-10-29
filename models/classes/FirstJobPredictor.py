import torch
import ipdb
import pytorch_lightning as pl
from models.classes import EncoderBiLSTM, DecoderLSTM, EncoderWithElmo, DecoderWithElmo
from torch.nn import functional as F


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, embeddings, hidden_size, num_layer, vector_size, output_size, MAX_SEQ_LENGTH, hparams):
        super().__init__()
        self.hp = hparams
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.embedding_layer = torch.nn.Embedding(embeddings.size(0), 100, padding_idx=0)
        self.embedding_layer.load_state_dict({'weight': embeddings[:, :100]})

        if self.hp.ft_type != "elmo":
            self.enc = EncoderBiLSTM(self.hp)
            self.dec = DecoderLSTM(self.hp)
        else:
            self.enc = EncoderBiLSTM(self.hp)
            self.dec = DecoderLSTM(self.hp)

    def forward(self, job_id, len_seq, enforce_sorted):
        ipdb.set_trace()

        return results, hidden

    def training_step(self, mini_batch, batch_nb):
        ipdb.set_trace()

    def validation_step(self, mini_batch, batch_nb):
        ipdb.set_trace()

    def validation_end(self, outputs):
        ipdb.set_trace()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def test_step(self, mini_batch, batch_idx):
        ipdb.set_trace()

    def test_epoch_end(self, outputs):
        ipdb.set_trace()
