import numpy as np
import torch
import random
import uuid
import os
import re
import time
import string
import argparse
import yaml
from collections import Counter
from os.path import join as pjoin


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix.copy()).type(torch.FloatTensor))


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        return 1


def max_len(list_of_list):
    return max(map(len, list_of_list))


def max_tensor_len(list_of_tensor, dim):
    tmp = []
    for t in list_of_tensor:
        tmp.append(t.size(dim))
    return max(tmp)


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    else:
        maxlen = max(np.max(lengths), maxlen)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def is_sub_list(long_list, short_list):
    # return True if the short list is a sublist of the long one
    if long_list is None or short_list is None:
        return False
    if len(long_list) == 0 or len(short_list) == 0:
        return False
    key = short_list[0]
    for i in range(len(long_list)):
        if long_list[i] == key:
            if long_list[i: i + len(short_list)] == short_list:
                return True
    return False


def get_sufficient_info_reward(observation_id_list, ground_truth_answer_ids):
    sufficient_info_reward = []

    for i in range(len(observation_id_list)):
        observation = observation_id_list[i]
        gt_answer = ground_truth_answer_ids[i][0]  # use the 1st answer
        if is_sub_list(observation, gt_answer):
            sufficient_info_reward.append(1.0)
        else:
            sufficient_info_reward.append(0.0)
    return np.array(sufficient_info_reward)


def get_qa_reward(pred_string, ground_truth_string, mode="f1"):
    qa_reward = []
    for i in range(len(pred_string)):
        if mode == "f1":
            qa_reward.append(f1_score_over_ground_truths(pred_string[i], ground_truth_string[i]))
        else:
            pred = normalize_string(pred_string[i])
            gt = [normalize_string(item) for item in ground_truth_string[i]]
            qa_reward.append(float(pred in gt))
    return np.array(qa_reward)


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def get_answer_strings(sentence_ids, head_and_tails, tokenizer, special_token_ids=[]):
    res = []
    for sent, ht in zip(sentence_ids, head_and_tails):
        # pad special tokens
        ids = [sent[0]] + sent + [sent[-1]]
        h, t = ht[0], ht[1]
        if h >= len(ids):
            h = len(ids) - 1
        if t >= len(ids):
            t = len(ids) - 1
        if h < t:
            words = ids[h: t + 1]
        else:
            words = ids[t: h + 1]
        words = [item for item in words if item not in special_token_ids]
        words = tokenizer.decode(words)
        res.append(words)
    return res
            

def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        text = " " + re.sub(r'(\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\])', ' ', text) + " "
        text = text.replace(" a ", " ").replace(" an ", " ").replace(" the ", " ")
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        res = ""
        for ch in text:
            if ch in exclude:
                res += " "
            else:
                res += ch
        return res

    return white_space_fix(remove_articles(remove_punc(s)))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

    
def f1_score_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = f1_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def list_of_token_list_to_char_input(list_of_token_list, char2id):
    batch_size = len(list_of_token_list)
    max_token_number = max_len(list_of_token_list)
    max_char_number = max([max_len(item) for item in list_of_token_list])
    if max_char_number < 6:
        max_char_number = 6
    res = np.zeros((batch_size, max_token_number, max_char_number), dtype='int32')
    for i in range(batch_size):
        for j in range(len(list_of_token_list[i])):
            for k in range(len(list_of_token_list[i][j])):
                res[i][j][k] = char2id[list_of_token_list[i][j][k]]
    return res


def list_of_word_id_list_to_char_input(list_of_word_id_list, id2word, char2id):
    res = []
    for i in range(len(list_of_word_id_list)):
        res.append([id2word[item] for item in list_of_word_id_list[i]])
    return list_of_token_list_to_char_input(res, char2id)


def get_answer_position(sentence_ids, answer_ids):
    res = []
    for i in range(len(sentence_ids)):
        if not is_sub_list(sentence_ids[i], answer_ids[i]):
            res.append(None)
        else:
            tmp = [0, 0]
            for j in range(len(sentence_ids[i])):
                if sentence_ids[i][j] != answer_ids[i][0]:
                    continue
                if sentence_ids[i][j: j + len(answer_ids[i])] == answer_ids[i]:
                    tmp = [j, j + len(answer_ids[i]) - 1]
                    tmp = [item + 1 for item in tmp]  # because a special token will be added in front of sentence.
                    break
            res.append(tmp)
    return res


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)

    def value(self, step):
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    return config


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


def get_stopword_ids(stopword_set, tokenizer):
        stopword_list = list(stopword_set)
        word2id = tokenizer.get_vocab()
        stopword_ids = set()
        for sw in stopword_list:
            if sw in word2id:
                stopword_ids.add(word2id[sw])
        return stopword_ids


def squeeze_last(t):
    if len(t.size()) == 2 and t.size()[-1] == 1:
        t = t.squeeze(-1)
    return t
