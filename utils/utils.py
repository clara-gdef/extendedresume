import re
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def word_seq_into_list(position, description):
    number_regex = re.compile(r'\d+(,\d+)?')
    new_tup = []
    whole_job = position.lower() + ' ' + description.lower()
    job = word_tokenize(whole_job)
    for tok in job:
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        else:
            new_tup.append(tok.lower())
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


def collate_for_flat_profiles(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    edu = [i[2] for i in batch]
    skills = [i[3] for i in batch]
    ind = [i[4] for i in batch]
    return ids, torch.from_numpy(np.stack(jobs)), torch.from_numpy(np.stack(edu)), skills, ind


def collate_for_edu(batch):
    ids = [i[0] for i in batch]
    edu = [i[1] for i in batch]
    skills = [i[2] for i in batch]
    ind = [i[3] for i in batch]
    return ids, torch.stack(edu), skills, ind


def get_model_params(args, dataset):
    return 300, args.hidden_size, len(dataset.skills_classes), len(dataset.ind_classes)


def test_for_skills(pred, labels, num_class):
    res = {}
    for threshold in np.linspace(0, 1, 10):
        new_preds = get_preds_wrt_threshold(pred, round(threshold, 1))
        res[round(threshold, 1)] = get_metrics(new_preds.squeeze(1).cpu().numpy(), labels.squeeze(1).cpu().numpy(), num_class, "skills")
        # res[round(threshold, 1) + "_@10"] = get_metrics(new_preds.cpu().numpy(), labels.cpu().numpy(), num_class, "skills")
    return res


def test_for_ind(pred, labels, num_class):
    predicted_classes = torch.argsort(pred, dim=-1, descending=True)
    res = {}
    res["ind"] = get_metrics(predicted_classes.squeeze(1).cpu().numpy()[:, 0], labels.squeeze(1).cpu().numpy(), num_class, "ind")
    res["ind_@10"] = get_metrics_at_k(predicted_classes.squeeze(1).cpu().numpy()[:, :10], labels.squeeze(1).cpu().numpy(), num_class, "ind")
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

