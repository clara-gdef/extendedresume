import pytorch_lightning as pl
import torch
import ipdb
from line_profiler import LineProfiler
import numpy as np
import pickle as pkl
import os
from utils.model import classes_to_one_hot, test_for_ind, test_for_skills
from models.classes.SkillsPredictor import SkillsPredictor
from models.classes.IndustryClassifier import IndustryClassifier


class EvalModels(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes_skills, num_classes_ind, datadir, hparams):
        super(EvalModels, self).__init__()
        self.hparams = hparams
        self.datadir = datadir
        self.num_classes_skills = num_classes_skills
        self.num_classes_ind = num_classes_ind

        self.test_pred_ind = []
        self.test_pred_skills = []
        self.test_label_ind = []
        self.test_label_skills = []
        self.test_label_fj = []
        self.test_label_fj = []

        self.skill_pred = SkillsPredictor(input_size, hidden_size, num_classes_skills)
        self.industry_classifier = IndustryClassifier(input_size, hidden_size, num_classes_ind)

    def forward(self, tmp_people):
        people = tmp_people.type(torch.FloatTensor).cuda()
        skills_pred = self.skill_pred.forward(people)
        ind_pred = self.industry_classifier.forward(people)

        return skills_pred, ind_pred

    def training_step(self, mini_batch, batch_nb):
        skills_pred, ind_pred = self.forward(mini_batch[1])
        lab_skills = mini_batch[-2]
        lab_ind = torch.LongTensor(mini_batch[-1]).cuda()
        lab_sk_1_hot = classes_to_one_hot(lab_skills, self.num_classes_skills)
        skills_loss = torch.nn.functional.binary_cross_entropy_with_logits(skills_pred, lab_sk_1_hot)
        ind_loss = torch.nn.functional.cross_entropy(ind_pred, lab_ind)
        loss = skills_loss + ind_loss
        tensorboard_logs = {"skills_loss": skills_loss, "ind_loss": ind_loss, 'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, mini_batch, batch_nb):
        skills_pred, ind_pred = self.forward(mini_batch[1])
        lab_skills = mini_batch[-2]
        lab_ind = torch.LongTensor(mini_batch[-1]).cuda()
        lab_sk_1_hot = classes_to_one_hot(lab_skills, self.num_classes_skills)
        skills_val_loss = torch.nn.functional.binary_cross_entropy_with_logits(skills_pred, lab_sk_1_hot)
        # ipdb.set_trace()
        ind_val_loss = torch.nn.functional.cross_entropy(ind_pred, lab_ind)
        val_loss = skills_val_loss + ind_val_loss
        # print(ind_val_loss)
        tensorboard_logs = {"skills_val_loss": skills_val_loss, "ind_val_loss": ind_val_loss, "val_loss": val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        val_losses = [i["val_loss"] for i in outputs]
        logs_skills_val_losses = [i["log"]["skills_val_loss"] for i in outputs]
        logs_ind_val_losses = [i["log"]["ind_val_loss"] for i in outputs]
        logs_val_losses = [i["log"]["val_loss"] for i in outputs]
        res_dict = {"val_loss": torch.mean(torch.stack(val_losses)),
                    "log": {
                        "skills_val_loss": torch.mean(torch.stack(logs_skills_val_losses)),
                        "ind_val_loss": torch.mean(torch.stack(logs_ind_val_losses)),
                        "val_loss": torch.mean(torch.stack(logs_val_losses)),
                    }}
        return res_dict
        # return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def test_step(self, mini_batch, batch_idx):
        skills_pred, ind_pred = self.forward(mini_batch[1])
        self.test_pred_skills.append(skills_pred)
        self.test_pred_ind.append(ind_pred)

        lab_skills = mini_batch[-2]
        lab_ind = torch.LongTensor(mini_batch[-1]).cuda()
        lab_sk_1_hot = classes_to_one_hot(lab_skills, self.num_classes_skills)
        self.test_label_ind.append(lab_ind)
        self.test_label_skills.append(lab_sk_1_hot)

    def test_epoch_end(self, outputs):
        skills_preds = torch.stack(self.test_pred_skills)
        skills_labels = torch.stack(self.test_label_skills)
        res_skills = test_for_skills(skills_preds, skills_labels, self.num_classes_skills)
        ind_preds = torch.stack(self.test_pred_ind)
        ind_labels = torch.stack(self.test_label_ind)
        res_ind = test_for_ind(ind_preds, ind_labels, self.num_classes_ind)
        print("Saving model outputs...")
        self.save_outputs()
        print("Model outputs saved!")
        return {**res_ind, **res_skills}

    def save_outputs(self):
        hp = self.hparams
        tgt_file = os.path.join(hp.datadir,
                                "outputs_eval_models_" + hp.model_type + "_" + hp.lr + "_" + hp.b_size + ".pkl")
        with open(tgt_file, "wb") as f:
            pkl.dump({"sk": {"preds": self.test_pred_skills, "labels": self.test_label_skills},
                      "ind": {"preds": self.test_pred_ind, "labels": self.test_label_ind}
                      }, f)
