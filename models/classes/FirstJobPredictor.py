import torch
import ipdb
import pytorch_lightning as pl
from models.classes import EncoderBiLSTM, DecoderLSTM, EncoderWithElmo, DecoderWithElmo
from torch.nn import functional as F


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, embeddings, datadir, index, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.index = index
        self.embedding_layer = torch.nn.Embedding(embeddings.size(0), 100, padding_idx=0)
        self.embedding_layer.load_state_dict({'weight': embeddings[:, :100]})

        if self.hp.ft_type != "elmo":
            self.enc = EncoderBiLSTM(embeddings[:, :100], self.hp)
            self.dec = DecoderLSTM(embeddings[:, :100], self.hp.hidden_size)
        else:
            self.enc = EncoderBiLSTM(self.hp)
            self.dec = DecoderLSTM(self.hp)

    def forward(self, job_id, len_seq):
        rep, att, hidden_state = self.enc.forward(job_id, len_seq, enforce_sorted=True)
        ipdb.set_trace()
        return rep, att, hidden_state

    def training_step(self, mini_batch, batch_nb):
        ipdb.set_trace()

    def validation_step(self, mini_batch, batch_nb):
        id, edu, edu_len, fj, fj_len = mini_batch
        results, attn, hidden = self.forward(edu, edu_len)
        # token =
        for step in range(len(fj_len)):
            ipdb.set_trace()

    def validation_end(self, outputs):
        ipdb.set_trace()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, mini_batch, batch_idx):
        ipdb.set_trace()

    def test_epoch_end(self, outputs):
        ipdb.set_trace()
