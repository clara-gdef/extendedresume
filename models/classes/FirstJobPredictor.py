import torch
import ipdb
import pytorch_lightning as pl
from models.classes import DecoderLSTM, DecoderWithElmo


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, embeddings, datadir, index, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.index = index
        self.embedding_layer = torch.nn.Embedding(embeddings.size(0), 100, padding_idx=0)
        self.embedding_layer.load_state_dict({'weight': embeddings[:, :100]})

        if self.hp.ft_type != "elmo":
            self.dec = DecoderLSTM(embeddings[:, :100], self.hp.hidden_size)
        else:
            self.dec = DecoderWithElmo(self.hp)
        self.decoded_tokens = []

    def forward(self, profile, fj):
        decoder_output, decoder_hidden = self.dec.forward(profile, fj[:, :-1])
        self.decoded_tokens.append(decoder_output.argmax(-1))
        return decoder_output

    def training_step(self, mini_batch, batch_nb):
        edu = mini_batch[1].unsqueeze(1)
        fj = mini_batch[-2]
        dec_outputs = self.forward(edu, fj)
        loss = torch.nn.functional.cross_entropy(dec_outputs, fj)
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, mini_batch, batch_nb):
        edu = mini_batch[1].unsqueeze(1)
        fj = mini_batch[-2]
        dec_outputs = self.forward(edu, fj)
        val_loss = torch.nn.functional.cross_entropy(dec_outputs, fj)
        tensorboard_logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, mini_batch, batch_idx):
        ipdb.set_trace()

    def test_epoch_end(self, outputs):
        ipdb.set_trace()
