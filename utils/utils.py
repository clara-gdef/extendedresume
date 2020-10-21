import re
from nltk.tokenize import word_tokenize


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
