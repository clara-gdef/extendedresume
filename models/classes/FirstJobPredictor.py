import os

import torch
import ipdb
import pytorch_lightning as pl
from models.classes import DecoderLSTM, DecoderWithElmo


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, dim, datadir, index, elmo, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.index = index

        if self.hp.ft_type != "elmo":
            self.dec = DecoderLSTM(dim, self.hp.hidden_size, len(index))
        else:
            self.dec = DecoderWithElmo(elmo,
                                       emb_dimension=1024,
                                       hidden_size=self.hp.hidden_size,
                                       num_layer=1,
                                       output_size=len(index))
        self.decoded_tokens = []
        self.decoded_tokens_test = []
        self.label_tokens_test = []

    def forward(self, profile, fj):
        decoder_output, decoder_hidden = self.dec.forward(profile, fj)
        self.decoded_tokens.append(decoder_output.argmax(-1))
        return decoder_output

    def training_step(self, mini_batch, batch_nb):
        dec_outputs = []
        tmp = 0
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            for num_tokens in range(fj.shape[1] - 1):
                dec_output = self.forward(edu, fj[:, num_tokens].unsqueeze(1))
                dec_outputs.append(dec_output)
                tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0)
            loss = tmp / (fj.shape[0] + fj.shape[1])
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj_lab)
        rev_index = {v: k for k, v in self.index.items()}
        ############
        outputs = torch.stack(dec_outputs).squeeze(2).transpose(1, 0)
        if batch_nb == 10:
            print("PREDICTION")
            pred = ""
            for w in outputs[-1]:
                word = torch.argmax(w)
                pred += rev_index[word.item()] + " "
            print(pred)
        # print("LABEL")
        # lab = ""
        # for w in fj[0]:
        #     lab += rev_index[w.item()] + " "
        # print(lab)
        ############
        tensorboard_logs = {'loss_CE': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, mini_batch, batch_nb):
        dec_outputs = []
        tmp = 0
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            for num_tokens in range(fj.shape[1] - 1):
                dec_output = self.forward(edu, fj[:, num_tokens].unsqueeze(1))
                dec_outputs.append(dec_output)
                ipdb.set_trace()
                tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0)
            val_loss = tmp / (fj.shape[0] + fj.shape[1])
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj)

        tensorboard_logs = {'val_CE': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, mini_batch, batch_idx):
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]

        token = self.index["SOD"]
        dec = []
        lab = []
        for i in range(len(fj[0])):
            tok_tensor = torch.LongTensor(1, 1)
            tok_tensor[:, 0] = token
            output, decoder_hidden = self.dec(edu, tok_tensor)
            dec_word = output.argmax(-1).item()
            dec.append(dec_word)
            lab.append(fj[0][i].item())
            token = dec_word
        self.decoded_tokens_test.append(dec)
        self.label_tokens_test.append(lab)

    def test_epoch_end(self, outputs):
        rev_index = {v: k for k, v in self.index.items()}

        pred_file = os.path.join(self.datadir, "pred_ft_" + self.hp.ft_type + ".txt")
        with open(pred_file, 'a') as f:
            for sentence in self.decoded_tokens_test:
                for w in sentence:
                    f.write(rev_index[w] + ' ')
                f.write("\n")

        lab_file = os.path.join(self.datadir, "label_ft_" + self.hp.ft_type + ".txt")
        with open(lab_file, 'a') as f:
            for sentence in self.label_tokens_test[1:]:
                for w in sentence:
                    f.write(rev_index[w] + ' ')
                f.write("\n")
        ipdb.set_trace()
        cmd_line = './multi-bleu.perl ' + lab_file + ' < ' + pred_file + ''
        os.system(cmd_line)