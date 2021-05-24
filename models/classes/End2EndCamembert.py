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
        # return loss

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
        pred_jobs = [self.tokenizer.decode(torch.stack(i).squeeze(-1).squeeze(-1), skip_special_tokens=True) for i in self.test_nj_pred]
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
            if self.hp.input_recopy == "True":
                if str_preds != labels:
                    ipdb.set_trace()
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

    def get_att_control_score(self, predicted_tokens, expected_exp, expected_ind):
        mod_type = self.hp.classifier_type
        predicted_inds, predicted_exps = [], []

        rev_ind_dict = dict()
        for k, v in self.industry_dict.items():
            if len(v.split(" ")) > 1:
                new_v = "_".join(v.split(" "))
            else:
                new_v = v
            rev_ind_dict[new_v] = k
        ind_labels = torch.stack(expected_ind).cpu().numpy()
        assert 0 <= min(ind_labels)
        assert len(self.industry_dict) > max(ind_labels)
        delta_labels = torch.stack(expected_exp).cpu().numpy()

        for sequence in tqdm(predicted_tokens, desc=f"evaluating attribute control with {mod_type} model..."):
            sentence = self.tokenizer.decode(sequence[0], skip_special_tokens=True)
            predictions_ind = handle_fb_preds(self.classifier_ind.predict(sentence, k=10))
            predictions_exp = handle_fb_preds(self.classifier_exp.predict(sentence, k=2))
            predicted_inds.append([rev_ind_dict[i] for i in predictions_ind])
            predicted_exps.append([int(i) for i in predictions_exp])
        predicted_exps = np.array(predicted_exps)
        predicted_inds = np.array(predicted_inds)
        preds_ind_at_1 = np.expand_dims(predicted_inds[:, 0], 1)
        preds_delta_at_1 = np.expand_dims(predicted_exps[:, 0], 1)

        ind_metrics = get_metrics(preds_ind_at_1, ind_labels, len(self.industry_dict), 'ind')
        delta_metrics = get_metrics(preds_delta_at_1, delta_labels, len(self.delta_dict), 'exp')

        if self.hp.print_cm == "True":
            print_cm(preds_ind_at_1, ind_labels, self.desc, self.industry_dict, "ind")
            print_cm(preds_delta_at_1, delta_labels, self.desc, self.delta_dict, "delta")

        ind_metrics_at10 = get_metrics_at_k(predicted_inds, ind_labels, len(self.industry_dict), 'ind@10')
        delta_metrics_at2 = get_metrics_at_k(predicted_exps, delta_labels, len(self.delta_dict), 'exp@2')
        return {**ind_metrics, **ind_metrics_at10}, {**delta_metrics,
                                                     **delta_metrics_at2}, predicted_inds, predicted_exps

    def get_ppl(self, decoded_tokens_indices):
        # for debug, don't put on cuda
        # self.ppl_model = self.ppl_model.cpu()
        ce = []
        for sentence in tqdm(decoded_tokens_indices, desc="computing perplexity..."):
            sentence_str = self.tokenizer.decode(sentence[0], skip_special_tokens=True)
            tmp = self.tokenizer(sentence_str, truncation=True, padding="max_length", max_length=self.max_len,
                                 return_tensors="pt")
            tokenized_sentence, mask = tmp["input_ids"].to(self.decoder.device), tmp["attention_mask"].to(
                self.decoder.device)
            outputs = self.ppl_model.ppl_model(tokenized_sentence, return_dict=True)
            logits = outputs.logits
            tmp2 = torch.nn.functional.cross_entropy(logits.transpose(-1, 1)[:, :, :-1], tokenized_sentence[:, 1:],
                                                     reduction='none')
            ce.append((tmp2 * mask[:, 1:]).sum() / mask[:, 1:].sum())
            # ipdb.set_trace()
        try:
            return 2 ** (torch.stack(ce).mean().item() / math.log(2))
        except OverflowError:
            return math.inf

    def add_noise(self, jobs, masks, max_len):
        sample_len = len(jobs)
        # https://arxiv.org/pdf/1711.00043.pdf p4, "noise model"
        if self.hp.drop_proba > 0:
            drop_proba = self.hp.drop_proba
            # word drop
            seq_len = [sum(i).item() for i in masks]
            keep = np.random.rand(sample_len, max_len) >= drop_proba
            keep[:, 0] = 1
            keep = torch.from_numpy(keep).type_as(jobs)
            dropped_jobs = torch.ones(sample_len, max_len).type_as(jobs)
            for i in range(max_len - 2):
                dropped_jobs[:, i] = keep[:, i] * jobs[:, i]
            for b in range(sample_len):
                dropped_jobs[b, seq_len[b] - 1] = 6
            dropped_jobs = dropped_jobs.type_as(jobs)
        else:
            dropped_jobs = jobs
        # word shuffle
        if self.hp.shuffle_dist > 0:
            shuffle_dist = self.hp.shuffle_dist
            shuffled_jobs = dropped_jobs.clone()
            permutation = np.arange(max_len)
            perm_tmp = torch.from_numpy(permutation).expand(sample_len, max_len)
            rdn_tensor = torch.zeros(sample_len, max_len)
            for i in range(max_len):
                rdn_tensor[:, i] = torch.from_numpy(np.random.uniform(0, shuffle_dist + 1, sample_len))
            perm_exp = perm_tmp + rdn_tensor
            perm_exp[:, 0] = -1
            new_indices = perm_exp.argsort()
            for b in range(sample_len):
                for i in range(max_len):
                    shuffled_jobs[b][i] = dropped_jobs[b][new_indices[b][i]]
            return shuffled_jobs.type_as(jobs)
        else:
            return dropped_jobs

    @staticmethod
    def get_prediction_mask(sentence):
        mask = torch.zeros(sentence.shape[0], sentence.shape[1])
        eos_flag = False
        for pos, token in enumerate(sentence[0]):
            if not eos_flag:
                mask[0, pos] = 1
            if token.item() == 6:  # token for eos
                eos_flag = True
        return mask.type(torch.cuda.LongTensor)

