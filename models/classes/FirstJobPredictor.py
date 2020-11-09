import os

import torch
import ipdb
import pytorch_lightning as pl
from models.classes import DecoderLSTM, DecoderWithElmo


class FirstJobPredictor(pl.LightningModule):
    def __init__(self, dim, datadir, index, elmo, class_weights, hidden_state, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.index = index
        self.hidden_state = hidden_state
        # # dirty trick : under weigh the "UNK" token class
        # class_weights = torch.ones(40005)
        # class_weights[4] = 10
        # self.class_weight = class_weights.cuda()

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

    def forward(self, profile, fj, hs):
        decoder_output, decoder_hidden = self.dec.forward(profile, fj, hs)
        return decoder_output, decoder_hidden

    def training_step(self, mini_batch, batch_nb):
        print("BATCH NUM : " + str(batch_nb))
        dec_outputs = []
        tmp = 0
        num_words = 0
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            num_words += sum(mini_batch[-1])
            dec_output, hs = self.forward(edu, fj[:, 1:].unsqueeze(1), self.hidden_state)
            self.hidden_state = hs
            dec_outputs.append(dec_output)
            tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, 1:], ignore_index=0,
                                                     reduction="sum")
            # for num_tokens in range(fj.shape[1] - 1):
            #     dec_output, hs = self.forward(edu, fj[:, num_tokens].unsqueeze(1), self.hidden_state)
            #     self.hidden_state = hs
            #     dec_outputs.append(dec_output)
            #     tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0, reduction="sum")
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj_lab)
        ############
        # rev_index = {v: k for k, v in self.index.items()}
        # ipdb.set_trace()
        # outputs = torch.stack(dec_outputs).squeeze(2).transpose(1, 0)
        # if batch_nb == 0:
        #     print("PREDICTION")
        #     pred = ""
        #     for w in outputs[-1]:
        #         word = torch.argmax(w, dim=-1)
        #         pred += rev_index[word.item()] + " "
        #     print(pred)
        #############
        loss = tmp / num_words
        self.log('loss_CE', loss)
        return {'loss': loss}

    def validation_step(self, mini_batch, batch_nb):
        dec_outputs = []
        tmp = 0
        num_words = 0
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            num_words += sum(mini_batch[-1])
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            num_words += sum(mini_batch[-1])
            dec_output, hs = self.forward(edu, fj[:, 1:].unsqueeze(1), self.hidden_state)
            self.hidden_state = hs
            dec_outputs.append(dec_output)
            tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, 1:], ignore_index=0,
            # for num_tokens in range(fj.shape[1] - 1):
            #     dec_output, hs = self.forward(edu, fj[:, num_tokens].unsqueeze(1), self.hidden_state)
            #     self.hidden_state = hs
            #     dec_outputs.append(dec_output)
            #     tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0, reduction="sum")
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj)

        val_loss = tmp / num_words
        self.log('val_loss_CE', val_loss)
        return {'val_loss': val_loss}


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
            dec_word = torch.argmax(output, dim=-1).item()
            dec.append(dec_word)
            lab.append(fj[0][i].item())
            token = dec_word
        self.decoded_tokens_test.append(dec)
        self.label_tokens_test.append(lab)

    def test_epoch_end(self, outputs):
        pred_file = os.path.join(self.datadir, "pred_ft_" + self.hp.ft_type + ".txt")
        lab_file = os.path.join(self.datadir, "label_ft_" + self.hp.ft_type + ".txt")
        if os.path.isfile(pred_file):
            os.system("rm " + pred_file)
            print("Removed previous pred file.")
        if os.path.isfile(lab_file):
            os.system("rm " + lab_file)
            print("Removed previous lable file.")

        rev_index = {v: k for k, v in self.index.items()}

        with open(pred_file, 'a') as f:
            for sentence in self.decoded_tokens_test:
                for w in sentence:
                    f.write(rev_index[w] + ' ')
                f.write("\n")

        with open(lab_file, 'a') as f:
            for sentence in self.label_tokens_test:
                for w in sentence:
                    f.write(rev_index[w] + ' ')
                f.write("\n")
        ipdb.set_trace()
        cmd_line = './multi-bleu.perl ' + lab_file + ' < ' + pred_file + ''
        os.system(cmd_line)
