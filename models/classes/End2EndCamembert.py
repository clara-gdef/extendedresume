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
from models.classes.FirstJobPredictor import FirstJobPredictor
from transformers import CamembertTokenizer, CamembertModel, CamembertForCausalLM


class End2EndCamembert(pl.LightningModule):
    def __init__(self, datadir, emb_size, desc, vocab_size, model_path, hp):
        super().__init__()
        self.datadir = datadir
        self.hp = hp
        self.emb_dim = emb_size
        self.desc = desc
        self.model_path = model_path
        self.voc_size = vocab_size
        self.max_len = hp.max_len

        with open(os.path.join(self.datadir, "good_skills.p"), 'rb') as f_name:
            self.skill_dict = pkl.load(f_name)
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

        self.num_ind = len(self.industry_dict)
        self.num_skills = len(self.skill_dict)

        self.encoder = CamembertModel.from_pretrained('camembert-base')
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.classifiers = EvalModels(input_size, hidden_size, num_classes_skills, num_classes_ind, datadir, hparams)
        self.job_generator = FirstJobPredictor(dim, datadir, index, elmo, class_weights, hparams)


    def forward(self, sentences, ind_indices, skills_indices, batch_nb):
        ipdb.set_trace()

    def inference(self, jobs, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices):
        ipdb.set_trace()

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
        self.test_decoded_outputs = []
        self.test_decoded_labels = []
        self.classifier_ind = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_ind_10.bin")
        self.classifier_exp = fasttext.load_model(f"/data/gainondefor/dynamics/ft_att_classifier_delta_5_10.bin")
        self.ppl_model = CausalLM(self.datadir, "causal_lm_bs100_lr1e-07_adam", self.voc_size, self.model_path,
                                  self.hp).to(self.decoder.device)
        self.ppl_model.is_decoder = True
        self.ppl_model.load_state_dict(
            torch.load("/data/gainondefor/dynamics/causal/lm/bs100/lr1e-07/adam/epoch=05.ckpt")["state_dict"])
        with open(os.path.join(self.datadir, "ind_class_dict.pkl"), 'rb') as f_name:
            self.industry_dict = pkl.load(f_name)

    def test_step(self, batch, batch_nb):
        sentences, ind_indices, delta_indices = batch[0], batch[1], batch[2]
        if self.hp.input_recopy == "True":
            self.test_decoded_labels.append((sentences, delta_indices, ind_indices))
        else:
            delta_tilde_indices, ind_tilde_indices = sample_y_tilde(delta_indices, ind_indices, len(self.delta_dict),
                                                                    len(self.industry_dict), self.hp.no_tilde)
            decoded_outputs = self.inference(sentences, delta_indices, ind_indices, delta_tilde_indices,
                                             ind_tilde_indices)
            self.test_decoded_outputs.append(decoded_outputs)
            self.test_decoded_labels.append(
                (sentences, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices))

    def test_epoch_end(self, outputs):
        print("Inference on testset completed. Commencing evaluation...")
        initial_jobs = [i[0] for i in self.test_decoded_labels]
        initial_exp = [i[2] for i in self.test_decoded_labels]
        initial_ind = [i[1] for i in self.test_decoded_labels]
        if self.hp.input_recopy == "True":
            str_list = [i[0] for i in initial_jobs]
            tokenized = self.tokenizer(str_list, truncation=True, padding="max_length", max_length=self.max_len,
                                       return_tensors="pt")
            ppl = self.get_ppl(tokenized["input_ids"])
            ind_metrics, delta_metrics, ind_preds, delta_preds = self.get_att_control_score(tokenized["input_ids"],
                                                                                            initial_exp, initial_ind)
            ipdb.set_trace()
            bleu = self.get_bleu_score(tokenized["input_ids"], initial_jobs)
        else:
            expected_delta = [i[3] for i in self.test_decoded_labels]
            expected_ind = [i[4] for i in self.test_decoded_labels]
            ## NORMAL EVAL
            ppl = self.get_ppl(self.test_decoded_outputs)
            stacked_outs = torch.stack(self.test_decoded_outputs)
            ind_metrics, delta_metrics, ind_preds, delta_preds = self.get_att_control_score(stacked_outs,
                                                                                            expected_delta,
                                                                                            expected_ind)
            bleu = prettify_bleu_score((self.get_bleu_score(stacked_outs.squeeze(1), initial_jobs)))
            # ipdb.set_trace()
        if self.hp.print_to_csv == "True":
            csv_file = print_tilde_to_csv(initial_jobs, initial_exp, initial_ind, self.test_decoded_outputs,
                                          expected_delta, expected_ind, delta_preds, ind_preds, self.desc,
                                          self.vocab_dict, self.industry_dict, self.delta_dict)
            df = pd.read_csv(csv_file)
            df.to_html(f'html/{self.desc}.html')
        print({"Avg ppl": ppl, **bleu, **ind_metrics, **delta_metrics})
        return {"Avg ppl": ppl, **bleu, **ind_metrics, **delta_metrics}

    def get_y_and_y_tilde(self, delta_indices, ind_indices, delta_tilde_indices, ind_tilde_indices):
        atts_as_first_token = embed_and_avg_attributes(delta_indices,
                                                       ind_indices,
                                                       self.attribute_embedder_as_tokens_exp.weight.shape[-1],
                                                       self.attribute_embedder_as_tokens_exp,
                                                       self.attribute_embedder_as_tokens_ind)
        atts_as_bias = embed_and_avg_attributes(delta_indices,
                                                ind_indices,
                                                self.voc_size,
                                                self.attribute_embedder_as_bias_exp,
                                                self.attribute_embedder_as_bias_ind)
        atts_tilde_as_first_token = embed_and_avg_attributes(delta_tilde_indices,
                                                             ind_tilde_indices,
                                                             self.attribute_embedder_as_tokens_exp.weight.shape[
                                                                 -1],
                                                             self.attribute_embedder_as_tokens_exp,
                                                             self.attribute_embedder_as_tokens_ind
                                                             )
        atts_tilde_as_bias = embed_and_avg_attributes(delta_tilde_indices,
                                                      ind_tilde_indices,
                                                      self.voc_size,
                                                      self.attribute_embedder_as_bias_exp,
                                                      self.attribute_embedder_as_bias_ind)
        return atts_as_first_token, atts_as_bias, atts_tilde_as_first_token, atts_tilde_as_bias

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
