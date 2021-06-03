import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import math
import pickle as pkl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import joblib
import fasttext
import pandas as pd
from models.classes.EvalModels import EvalModels
from models.classes.FirstJobPredictorForCamembert import FirstJobPredictorForCamembert
from transformers import CamembertTokenizer, CamembertModel, CamembertForCausalLM

from utils import classes_to_one_hot, prettify_bleu_score, compute_bleu_score, test_for_skills, test_for_ind


class End2EndCamembert(pl.LightningModule):
    def __init__(self, datadir, desc, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.desc = desc
        self.model_path = model_path
        self.max_len = hp.max_len

        with open(os.path.join(self.datadir, "good_skills.p"), 'rb') as f_name:
            self.skill_dict = pkl.load(f_name)
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

        self.num_ind = len(self.industry_dict)
        self.num_skills = len(self.skill_dict)

        self.encoder = CamembertModel.from_pretrained('camembert-base')
        if self.hp.end2end != "True":
            self.encoder.requires_grad_ = False
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.emb_dim = self.encoder.embeddings.word_embeddings.embedding_dim
        self.voc_size = self.encoder.embeddings.word_embeddings.num_embeddings

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.classifiers = EvalModels(input_size=self.emb_dim,
                                      hidden_size=self.hp.hidden_size,
                                      num_classes_skills=self.num_skills,
                                      num_classes_ind=self.num_ind,
                                      datadir=datadir,
                                      hparams=hp)
        self.job_generator = FirstJobPredictorForCamembert(dim=self.emb_dim,
                                                           datadir=datadir,
                                                           tokenizer=self.tokenizer,
                                                           hparams=hp)

    def forward(self, sentences, ind_indices, skills_indices, batch_nb):
        # build prof
        exp_len = []
        flattened_sentences = []
        jobs_to_generate = []
        for prof in sentences:
            # we skip the first exp as it is the label for job prediction
            jobs_to_generate.append(prof[0])
            for exp in prof[1:]:
                flattened_sentences.append(exp)
            exp_len.append(len(prof)-1)
        inputs = self.tokenizer(flattened_sentences, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']
        # avg
        reshaped_profiles = torch.zeros(self.hp.b_size, self.max_len, self.emb_dim)
        start = 0
        for num_prof, length in enumerate(exp_len):
            end = start + length
            reshaped_profiles[num_prof] = torch.mean(encoder_outputs[start:end], dim=0)
            start = end
        if self.hp.end2end != "True":
            reshaped_profiles.detach()

        # pred skills & pred ind
        lab_sk_1_hot = classes_to_one_hot(skills_indices, self.num_skills)
        # we use the encoder's last hidden state
        pred_sk, pred_ind = self.classifiers(reshaped_profiles[:, -1, :])
        loss_sk = torch.nn.functional.binary_cross_entropy_with_logits(pred_sk,
                                                                       lab_sk_1_hot, reduction="mean")
        loss_ind = torch.nn.functional.cross_entropy(pred_ind,
                                                    torch.LongTensor(ind_indices).cuda(), reduction="mean")

        # gen next job
        inputs = self.tokenizer(jobs_to_generate, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        jobs_tokenized = inputs["input_ids"].cuda()
        jobs_to_generate_embedded = self.encoder.embeddings(jobs_tokenized)
        loss_nj, decoder_output, decoder_hidden = self.job_generator.forward(reshaped_profiles.cuda(), jobs_to_generate_embedded, jobs_tokenized)

        if self.hp.print_preds == "True" and batch_nb == 0:
            tmp = torch.argmax(decoder_output[0], dim=-1)
            pred = self.tokenizer.decode(tmp, skip_special_tokens=True)
            label = self.tokenizer.decode(jobs_tokenized[0], skip_special_tokens=True)
            print("==================================================")

            print("LABEL : " + label)
            print("PRED : " + pred)
            from sklearn.metrics import hamming_loss
            from utils.model import get_preds_wrt_threshold, get_metrics_for_skills

            tmp_sk_label = lab_sk_1_hot.clone().cpu().numpy()
            tmp_sk_pred = get_preds_wrt_threshold(pred_sk, .5)
            print("==================================================")
            sk_met = get_metrics_for_skills(tmp_sk_label, tmp_sk_pred, 523, "skills")
            print(f"METRICS SKILLS : {sk_met}")
            print("==================================================")
            print(f"LABEL INDUSTRY : {ind_indices[0]}")
            print(f"PRED INDUSTRY : {torch.argmax(pred_ind[0], dim=-1)}")
        # return loss
        # loss_total = loss_nj / sum(sum(mask))

        loss_total = loss_sk + loss_ind + (loss_nj / sum(sum(mask)))

        if torch.isnan(loss_total):
            ipdb.set_trace()
        if torch.isinf(loss_total):
            ipdb.set_trace()
        return loss_total

    def inference(self, sentences):
        # build prof
        exp_len = []
        flattened_sentences = []
        jobs_to_generate = []
        for prof in sentences:
            # we skip the first exp as it is the label for job prediction
            jobs_to_generate.append(prof[0])
            for exp in prof[1:]:
                flattened_sentences.append(exp)
            exp_len.append(len(prof) - 1)
        inputs = self.tokenizer(flattened_sentences, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        input_tokenized, mask = inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()
        encoder_outputs = self.encoder(input_tokenized, mask)['last_hidden_state']
        # avg
        reshaped_profiles = torch.zeros(1, self.max_len, self.emb_dim)
        start = 0
        for num_prof, length in enumerate(exp_len):
            end = start + length
            reshaped_profiles[num_prof] = torch.mean(encoder_outputs[start:end], dim=0)
            start = end
        # pred skills & pred ind
        # we use the encoder's last hidden state
        pred_sk, pred_ind = self.classifiers(reshaped_profiles[:, -1, :])
        # gen next job
        inputs = self.tokenizer(jobs_to_generate, truncation=True, padding="max_length", max_length=self.max_len,
                                return_tensors="pt")
        jobs_tokenized = inputs["input_ids"].cuda()
        jobs_to_generate_embedded = self.encoder.embeddings(jobs_tokenized)
        decoded_tokens, _ = self.job_generator.inference(reshaped_profiles.cuda(), jobs_to_generate_embedded, self.encoder.embeddings)
        return pred_sk, pred_ind, decoded_tokens

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.hp.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr)

    def training_step(self, batch, batch_nb):
        ids, sentences, skills_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        loss = self.forward(sentences, ind_indices, skills_indices, batch_nb)
        self.log('train_loss_ep', loss, on_step=False, on_epoch=True)
        self.log('train_loss_st', loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        ids, sentences, skills_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        val_loss = self.forward(sentences, ind_indices, skills_indices, batch_nb)
        self.log('val_loss', val_loss)
        return {"val_loss": val_loss}

    def on_test_epoch_start(self):
        self.test_nj_pred = []
        self.test_nj_labs = []
        self.test_sk_pred = []
        self.test_sk_labs = []
        self.test_ind_pred = []
        self.test_ind_labs = []

    def test_step(self, batch, batch_nb):
        ids, sentences, skills_indices, ind_indices = batch[0], batch[1], batch[2], batch[3]
        pred_sk, pred_ind, decoded_tokens = self.inference(sentences)
        self.test_nj_pred.append(decoded_tokens)
        self.test_nj_labs.append(sentences[0])
        self.test_sk_pred.append(pred_sk[0])
        self.test_sk_labs.append(skills_indices[0])
        self.test_ind_pred.append(pred_ind[0])
        self.test_ind_labs.append(ind_indices[0])

    def test_epoch_end(self, outputs):
        print("Inference on testset completed. Commencing evaluation...")
        # get skill metrics
        skills_preds = torch.stack(self.test_sk_pred)
        skills_labels = classes_to_one_hot(self.test_sk_labs, self.num_skills)
        res_skills = test_for_skills(skills_preds, skills_labels, self.num_skills)
        # get ind_metrics
        ind_preds = torch.stack(self.test_ind_pred)
        lab_ind = torch.LongTensor(self.test_ind_labs)
        res_ind = test_for_ind(ind_preds.unsqueeze(1), lab_ind.unsqueeze(1), self.num_ind)
        # get bleu score
        pred_jobs = [torch.stack(i).squeeze(-1).squeeze(-1) for i in self.test_nj_pred]
        actual_jobs = self.test_nj_labs
        bleu = prettify_bleu_score((self.get_bleu_score(pred_jobs, actual_jobs)))
        print({**bleu, **res_ind, **res_skills})
        ipdb.set_trace()
        return {**bleu, **res_ind, **res_skills}

    def save_at_step(self, batch_nb):
        if not os.path.isdir(self.model_path):
            os.system("mkdir -p " + self.model_path)
        prev_file = os.path.join(self.model_path, "step_" + str(batch_nb - self.save_step) + "_save.ckpt")
        if os.path.isfile(prev_file):
            os.system('rm ' + prev_file)
        tgt_file = os.path.join(self.model_path, "step_" + str(batch_nb) + "_save.ckpt")
        if not os.path.isfile(tgt_file):
            torch.save(self, tgt_file)
            print("checkpoint saved @ " + str(tgt_file))

    def get_bleu_score(self, test_outputs, test_labels):
        lab_file = os.path.join(self.datadir, self.desc + '_bleu_LABELS.txt')
        pred_file = os.path.join(self.datadir, self.desc + '_bleu_PREDS.txt')
        if os.path.isfile(lab_file):
            os.system('rm ' + lab_file)
            print("Removed " + lab_file)
        if os.path.isfile(pred_file):
            os.system('rm ' + pred_file)
            print("Removed " + pred_file)
        # make bleu label file
        for num, predicted_tokens in tqdm(enumerate(test_outputs), desc="Building txt files for BLEU score..."):
            str_preds = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            labels = self.tokenizer.decode(
                self.tokenizer(test_labels[num][0], truncation=True, padding="max_length", max_length=self.max_len,
                               return_tensors="pt")["input_ids"][0], skip_special_tokens=True)
            if len(str_preds.split(" ")) < 1:
                ipdb.set_trace()
            if len(labels.split(" ")) < 1:
                ipdb.set_trace()
            with open(pred_file, "a+") as f:
                f.write(str_preds + " \n")
            with open(lab_file, "a+") as f:
                f.write(labels + " \n")
            # get score
        print(f"Computing BLEU score for model {self.desc}")
        print(f"Pred file: {pred_file}")
        return compute_bleu_score(pred_file, lab_file)
