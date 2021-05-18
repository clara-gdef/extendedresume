import glob

import numpy as np
import torch
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
from utils.pre_processing import word_list_to_indices


def collate_for_flat_profiles(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    edu = [i[2] for i in batch]
    skills = [i[3] for i in batch]
    ind = [i[4] for i in batch]
    return ids, torch.from_numpy(np.stack(jobs)), torch.from_numpy(np.stack(edu)), skills, ind


def collate_for_bert_edu(batch):
    ids = [i[0] for i in batch]
    edu = [i[2] for i in batch]
    skills = [i[3] for i in batch]
    ind = [i[4] for i in batch]
    return ids, edu, skills, ind


def collate_for_bert_jobs(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    skills = [i[3] for i in batch]
    ind = [i[4] for i in batch]
    return ids, jobs, skills, ind


def collate_for_edu(batch):
    ids = [i[0] for i in batch]
    edu = [i[1] for i in batch]
    skills = [i[2] for i in batch]
    ind = [i[3] for i in batch]
    return ids, torch.stack(edu), skills, ind


def collate_for_text_gen(batch):
    ids = [i[0] for i in batch]
    edu = [i[1] for i in batch]
    fj = [torch.LongTensor(i[2]) for i in batch]
    len_fj = [i[3] for i in batch]
    return ids, torch.stack(edu), torch.stack(fj), len_fj


def collate_for_text_gen_elmo(batch):
    ids = [i[0] for i in batch]
    edu = [i[1] for i in batch]
    fj = [i[2] for i in batch]
    fj_len = [i[3] for i in batch]
    fj_indices = [i[4] for i in batch]
    return ids, torch.stack(edu), fj, fj_len, torch.stack(fj_indices)


def get_model_params(args, dataset):
    if args.ft_type == "elmo":
        dim = 1024
    else:
        dim = 300
    return dim, args.hidden_size, len(dataset.skills_classes), len(dataset.ind_classes)


def test_for_skills(pred, labels, num_class):
    res = {}
    for threshold in tqdm(np.linspace(0, 1, 10)):
        print('Testing skills for threshold ' + str(round(threshold, 1)) + ' ...')
        new_preds = get_preds_wrt_threshold(pred, round(threshold, 1))
        res[round(threshold, 1)] = get_metrics(new_preds.squeeze(1).cpu().numpy(), labels.squeeze(1).cpu().numpy(), num_class, "skills")
        # res[round(threshold, 1) + "_@10"] = get_metrics(new_preds.cpu().numpy(), labels.cpu().numpy(), num_class, "skills")
    return res


def test_for_ind(pred, labels, num_class):
    print("Testing for industries")
    predicted_classes = torch.argsort(pred, dim=-1, descending=True)
    res = {}
    res["ind"] = get_metrics(predicted_classes.squeeze(1).cpu().numpy()[:, 0], labels.squeeze(1).cpu().numpy(), num_class, "ind")
    res["ind_@10"] = get_metrics_at_k(predicted_classes.squeeze(1).cpu().numpy()[:, :10], labels.squeeze(1).cpu().numpy(), num_class, "ind")
    print("Industries tested.")
    return res


def classes_to_one_hot(lab_skills, num_classes):
    new_labels = torch.zeros(len(lab_skills), num_classes)
    for person in range(len(lab_skills)):
        for sk in lab_skills[person]:
            new_labels[person, sk] = 1.
    return new_labels.cuda()


def get_metrics(preds, labels, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_metrics_for_skills(preds, labels, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "hamming_" + handle: hamming_loss(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_metrics_at_k(predictions, labels, num_classes, handle):
    out_predictions = []
    for index, pred in enumerate(predictions):
        if labels[index].item() in pred:
            out_predictions.append(labels[index].item())
        else:
            if type(pred[0]) == torch.Tensor:
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle)


def get_preds_wrt_threshold(pred, th):
    preds = []
    for person in pred:
        preds.append(((person > th).float()*1).type(torch.uint8))
    pred = torch.stack(preds).type(torch.FloatTensor)
    return pred


def get_latest_model(modeldir, xp_title):
    model_path = os.path.join(modeldir, xp_title)
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    return latest_file
