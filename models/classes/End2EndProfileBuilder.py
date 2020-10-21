import pytorch_lightning as pl
import torch
import ipdb
import numpy as np

from models.classes.SkillsPredictor import SkillsPredictor
from models.classes.IndustryClassifier import IndustryClassifier


class End2EndProfileBuilder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes_skills, num_classes_ind, hparams):
        super(End2EndProfileBuilder, self).__init__()
        self.hparams = hparams

        self.atn_layer = torch.nn.Linear(input_size, 1)
        self.skill_pred = SkillsPredictor(input_size, hidden_size, num_classes_skills)
        self.industry_classifier = IndustryClassifier(input_size, hidden_size, num_classes_ind)

    def forward(self, tmp_people):
        people = torch.from_numpy(np.stack(tmp_people)).type(torch.FloatTensor).cuda()
        atn = self.atn_layer(people)
        normed_atn = atn.clone()
        for ind, sample in enumerate(atn):
            normed_atn[ind] = torch.softmax(atn[ind], dim=0)
        new_people = self.ponderate_jobs(people, normed_atn)

        skills_pred = self.skill_pred.forward(new_people)
        ind_pred = self.industry_classifier.forward(new_people)

        return skills_pred, ind_pred

    def training_step(self, mini_batch, batch_nb):
        ipdb.set_trace()
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, mini_batch, batch_nb):
        ipdb.set_trace()
        tensorboard_logs = {**res_dict, 'val_loss': val_loss}
        return {'loss': val_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)


    def test_step(self, mini_batch, batch_idx):
        ipdb.set_trace()

    def test_epoch_end(self, outputs):
        ipdb.set_trace()