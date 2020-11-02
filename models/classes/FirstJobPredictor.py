import os

import torch
import ipdb
import pytorch_lightning as pl
from models.classes import DecoderLSTM, DecoderWithElmo


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, dim, datadir, index, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.index = index

        if self.hp.ft_type != "elmo":
            self.dec = DecoderLSTM(dim, self.hp.hidden_size, len(index))
        else:
            self.dec = DecoderWithElmo(self.hp)
        self.decoded_tokens = []
        self.decoded_tokens_test = []
        self.label_tokens_test = []

    def forward(self, profile, fj):
        decoder_output, decoder_hidden = self.dec.forward(profile, fj[:, :-1])
        self.decoded_tokens.append(decoder_output.argmax(-1))
        return decoder_output

    def training_step(self, mini_batch, batch_nb):
        edu = mini_batch[1].unsqueeze(1)
        fj = mini_batch[-2]
        dec_outputs = self.forward(edu, fj)
        loss = torch.nn.functional.cross_entropy(dec_outputs.transpose(2, 1), fj[:, :-1])
        tensorboard_logs = {'loss_CE': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, mini_batch, batch_nb):
        edu = mini_batch[1].unsqueeze(1)
        fj = mini_batch[-2]
        dec_outputs = self.forward(edu, fj)
        val_loss = torch.nn.functional.cross_entropy(dec_outputs.transpose(2, 1), fj[:, :-1])
        tensorboard_logs = {'val_CE': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        ipdb.set_trace()
        return outputs[-1]

    def training_epoch_end(self, outputs):
        ipdb.set_trace()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, mini_batch, batch_idx):
        edu = mini_batch[1].unsqueeze(1)
        fj = mini_batch[-2]
        token = self.index["SOD"]
        for i in range(len(fj[0])):
            tok_tensor = torch.LongTensor(1, 1)
            tok_tensor[:, 0] = token
            output, decoder_hidden = self.dec(edu, tok_tensor)
            dec_word = output.argmax(-1).item()
            self.decoded_tokens_test.append(dec_word)
            self.label_tokens_test.append(fj[0][i])
            token = dec_word
        ipdb.set_trace()

    def test_epoch_end(self, outputs):
        rev_index = {v: k for k, v in self.index.items()}

        pred_file = os.path.join(self.datadir, "pred_ft_" + self.ft_type + ".txt")
        with open(pred_file, 'a') as f:
            for w in self.decoded_tokens_test:
                f.write(rev_index[w] + ' ')
            f.write("\n")

        lab_file = os.path.join(self.datadir, "label_ft_" + self.ft_type + ".txt")
        with open(lab_file, 'a') as f:
            for w in self.label_tokens_test[1:]:
                f.write(rev_index[w] + ' ')
            f.write("\n")
        ipdb.set_trace()
