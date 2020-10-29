import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import ipdb
from allennlp.modules.elmo import batch_to_ids


class EncoderWithElmo(pl.LightningModule):
    def __init__(self, embeddings, vector_size, MAX_CAREER_LENGTH, hparams):
        super().__init__()

        self.hparams = hparams
        ipdb.set_trace()

    def word_level(self, sentences):
        character_ids = batch_to_ids(sentences)

        emb = self.elmo(character_ids.cuda())

        emb_tensor = emb["elmo_representations"][-1]

        # mask = emb["mask"]
        # self.alpha = self.masked_softmax(self.context(emb_tensor).squeeze(-1), mask)
        # sentence_rep = torch.einsum("blf,bl->bf", emb_tensor, self.alpha)

        return emb_tensor

    def forward(self, job_id):
        """len_seq : longueur effective des séquences """
        ipdb.set_trace()
        job_rep = self.word_level(job_id)
        return job_rep

