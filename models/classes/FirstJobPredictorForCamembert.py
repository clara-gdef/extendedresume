import os
import random
import torch
import ipdb
import pytorch_lightning as pl
from models.classes import DecoderLSTMForCamembert


class FirstJobPredictorForCamembert(pl.LightningModule):
    def __init__(self, dim, datadir, tokenizer, hparams):
        super().__init__()
        self.datadir = datadir
        self.hp = hparams
        self.tokenizer = tokenizer

        self.decoder = DecoderLSTMForCamembert(input_size=dim,
                                           hidden_size=self.hp.hidden_size,
                                           emb_dim=dim,
                                           out_size=tokenizer.vocab_size)

        self.decoded_tokens = []
        self.decoded_tokens_test = []
        self.label_tokens_test = []

    def forward(self, encoder_outputs, jobs_embedded, input_tokenized):
        h0 = (torch.zeros(1, self.hp.b_size, self.hp.hidden_size).type_as(jobs_embedded),
              torch.zeros(1, self.hp.b_size, self.hp.hidden_size).type_as(jobs_embedded))
        decoder_output, decoder_hidden = self.decoder.forward(encoder_outputs[:, :-1, :],
                                                              jobs_embedded[:, :-1, :],
                                                              h0)
        loss = torch.nn.functional.cross_entropy(decoder_output.transpose(-1, 1),
                                                 input_tokenized[:, 1:], reduction="sum", ignore_index=1)
        return loss, decoder_output, decoder_hidden

    def inference(self, encoder_outputs, jobs_embedded, embedder):
        decoded_tokens, posteriors = [], []
        sos = self.tokenizer.special_tokens_map["bos_token"]
        tkized_sos = self.tokenizer(sos, truncation=True, padding=True, max_length=len(jobs_embedded[0]),
                                return_tensors="pt")
        input_tokenized, mask = tkized_sos["input_ids"].cuda(), tkized_sos["attention_mask"].cuda()
        previous_token = embedder(input_tokenized)[0][0].unsqueeze(0).unsqueeze(1)
        prev_hidden = (torch.zeros(1, 1, self.hp.hidden_size).type_as(jobs_embedded),
              torch.zeros(1, 1, self.hp.hidden_size).type_as(jobs_embedded))
        for di in range(len(jobs_embedded[0]) - 1):
            decoder_output, decoder_hidden = self.decoder.forward(encoder_outputs[:, di, :].unsqueeze(1),
                                                              previous_token,
                                                              prev_hidden)
            posteriors.append(decoder_output)
            decoder_tok = torch.argmax(decoder_output, dim=-1)
            decoded_tokens.append(decoder_tok)
            previous_token = embedder(decoder_tok)
            prev_hidden = decoder_hidden
        return decoded_tokens, posteriors

    def training_step(self, mini_batch, batch_nb):
        dec_outputs = []
        tmp = 0
        num_words = 0
        hs = (torch.zeros(1, self.hp.b_size, self.hp.hidden_size).cuda(),
              torch.zeros(1, self.hp.b_size, self.hp.hidden_size).cuda())
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            num_words += sum(mini_batch[-1])
            # init first tokens
            tokens = torch.LongTensor([self.index["SOT"]]).expand(self.hp.b_size).unsqueeze(1)
            for num_tokens in range(fj.shape[1] - 1):
                # get pred
                dec_output, hs = self.forward(edu, tokens, hs)
                dec_outputs.append(dec_output)
                # if within teacher forcing range, next tokens will be the label ones
                if random.random() <= self.hp.tf:
                    tokens = fj[:, num_tokens].unsqueeze(1)
                # else, next tokens will be the predicted ones
                else:
                    tokens = dec_output.argmax(-1)
                tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0)
            loss = tmp / num_words
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj_lab)
        rev_index = {v: k for k, v in self.index.items()}
        ###########
        outputs = torch.stack(dec_outputs).squeeze(2).transpose(1, 0)
        if batch_nb == 0:
            print("PREDICTION")
            pred = ""
            for w in outputs[-1]:
                word = torch.argmax(w)
                pred += rev_index[word.item()] + " "
            print(pred)
            print("LABEL")
            lab = ""
            for w in fj[0]:
                lab += rev_index[w.item()] + " "
            print(lab)
            ##########
        self.log('loss_CE', loss)
        return {'loss': loss}

    def validation_step(self, mini_batch, batch_nb):
        dec_outputs = []
        tmp = 0
        num_words = 0
        hs = (torch.zeros(1, self.hp.b_size, self.hp.hidden_size).cuda(),
              torch.zeros(1, self.hp.b_size, self.hp.hidden_size).cuda())
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
            num_words += sum(mini_batch[-1])
            # init first tokens
            tokens = torch.LongTensor([self.index["SOT"]]).expand(self.hp.b_size).unsqueeze(1)
            for num_tokens in range(fj.shape[1] - 1):
                # get pred
                dec_output, hs = self.forward(edu, tokens, hs)
                dec_outputs.append(dec_output)
                tokens = dec_output.argmax(-1)
                tmp += torch.nn.functional.cross_entropy(dec_output.squeeze(1), fj[:, num_tokens], ignore_index=0)
            # for num_tokens in range(fj.shape[1] - 1):
            #     dec_output, hs = self.forward(edu, fj[:, num_tokens].unsqueeze(1), hs)
            #     tmp.append(dec_output.argmax(-1).item())
            #     dec_outputs.append(dec_output)
            #
            val_loss = tmp / num_words
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
            fj_lab = mini_batch[-1][:, 1:]
            dec_outputs = self.forward(edu, fj)

        self.log('val_CE', val_loss)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, mini_batch, batch_idx):
        if self.hp.ft_type != "elmo":
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[-2]
        else:
            edu = mini_batch[1].unsqueeze(1)
            fj = mini_batch[2]
        hs = (torch.zeros(1, 1, self.hp.hidden_size).cuda(),
              torch.zeros(1, 1, self.hp.hidden_size).cuda())
        token = self.index["SOT"]
        dec = []
        lab = []
        tok_tensor = torch.LongTensor(1, 1)
        # ipdb.set_trace()
        for i in range(fj.shape[1] - 1):
            tok_tensor[0, 0] = token
            output, hs = self.dec(edu, tok_tensor, hs)
            dec_word = output.argmax(-1).item()
            dec.append(dec_word)
            lab.append(fj[0][i].item())
            token = dec_word
        self.decoded_tokens_test.append(dec)
        self.label_tokens_test.append(lab)

    def test_epoch_end(self, outputs):
        rev_index = {v: k for k, v in self.index.items()}

        desc = self.hp.ft_type + '_' + str(self.hp.lr) + '_' + str(self.hp.b_size)
        pred_file = os.path.join(self.datadir, "pred_ft_" + desc + ".txt")
        lab_file = os.path.join(self.datadir, "label_ft_" + desc + ".txt")

        if os.path.isfile(lab_file):
            os.system('rm ' + lab_file)
        if os.path.isfile(pred_file):
            os.system('rm ' + pred_file)

        with open(pred_file, 'a') as f:
            for sentence in self.decoded_tokens_test:
                for w in sentence:
                    if rev_index[w] != "EOD":
                        f.write(rev_index[w] + ' ')
                    else:
                        break
                f.write("\n")

        with open(lab_file, 'a') as f:
            for sentence in self.label_tokens_test[1:]:
                for w in sentence:
                    if rev_index[w] != "EOD":
                        f.write(rev_index[w] + ' ')
                    else:
                        break
                f.write("\n")
        #cmd_line = './multi-bleu.perl ' + lab_file + ' < ' + pred_file + ' >> bleu_scores_' + desc +  '.txt'
        cmd_line = './multi-bleu.perl ' + lab_file + ' < ' + pred_file
        ipdb.set_trace()
        os.system(cmd_line)
