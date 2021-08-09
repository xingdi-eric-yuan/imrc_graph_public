import os
import re
import json
import codecs
import operator
import logging
from tqdm import tqdm

import numpy as np
import gym
from transformers import BertTokenizer
from generic import is_sub_list
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class GamifiedSquad(gym.Env):

    def __init__(self, config):
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.word2id = self.bert_tokenizer.get_vocab()
        self.has_token_set = set()
        # load dataset from file
        self.dataset = dict(
            train=dict(
                sentence_id_list=[], question_id=[], answer_string_list=[], answer_id_list=[], srl_list=[], question_srl_list=[]),
            valid=dict(
                sentence_id_list=[], question_id=[], answer_string_list=[], answer_id_list=[], srl_list=[], question_srl_list=[]),
            test=dict(
                sentence_id_list=[], question_id=[], answer_string_list=[], answer_id_list=[], srl_list=[], question_srl_list=[]),
        )
        self.tmp_vocab = {}
        for split in ["train", "valid", "test"]:
            self.load_dataset(split)
        print("loaded dataset from %s ..." % self.data_path)
        self.train_size = len(self.dataset["train"]["question_id"])
        self.valid_size = len(self.dataset["valid"]["question_id"])
        self.test_size = len(self.dataset["test"]["question_id"])
        self.batch_pointer, self.step_counter = None, None
        self.train_batch_pointer = 0
        self.data_size, self.batch_size, self.data, self.infos = None, None, None, None
        self.current_story = None
        self.current_answers, self.last_actions = None, None
        self.current_sentence_id_list, self.current_question_id, self.current_answer_id_list = None, None, None
        self.current_srl_list, self.current_question_srl_list = None, None
        self.split = "train"

    def load_dataset(self, split):
        
        file_path = os.path.join(self.data_path, "processed_squad.1.1.split." + split + ".json")
        data = json.load(codecs.open(file_path, 'r', 'utf-8'))
        data = data["data"]
        for i in tqdm(range(len(data))):
            sentence_list = data[i]["sents"]
            srl_list = data[i]["srls"]
            question = data[i]["q"]
            answer_id_list = data[i]["a"]
            question_srl_list = data[i]["q_srls"]

            has_answer = 0
            for a in answer_id_list:
                for s in sentence_list:
                    if is_sub_list(s, a):
                        has_answer = 1
                        break
                if has_answer == 1:
                    break
            if has_answer == 0:
                continue

            for sent in sentence_list:
                for w in sent:
                    self.has_token_set.add(w)
            for w in question:
                self.has_token_set.add(w)

            self.dataset[split]["sentence_id_list"].append(sentence_list)
            self.dataset[split]["question_id"].append(question)
            self.dataset[split]["answer_id_list"].append(answer_id_list)
            decoded_answers = [self.bert_tokenizer.decode(item) for item in answer_id_list]
            self.dataset[split]["answer_string_list"].append(decoded_answers)
            self.dataset[split]["srl_list"].append(srl_list)
            self.dataset[split]["question_srl_list"].append(question_srl_list)

            if self.debug_mode and len(self.dataset[split]["question_id"]) > 1000:
                break

    def read_config(self):
        self.data_path = self.config["general"]["data_path"]
        self.debug_mode = self.config["general"]["debug_mode"]
        if self.config["general"]["philly"]:
            self.data_path = os.environ['PT_DATA_DIR'] + "/" + self.data_path
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]
        self.start_from_beginning = self.config["general"]["start_from_beginning"]

        self.training_batch_size = self.config["training"]["batch_size"]
        self.max_nb_steps_per_episode = self.config["training"]["max_nb_steps_per_episode"]
        self.evaluate_batch_size = self.config["evaluate"]["batch_size"]

    def split_reset(self, split):
        if split == "train":
            self.data_size = self.train_size
            self.batch_size = self.training_batch_size
        elif split == "valid":
            self.data_size = self.valid_size
            self.batch_size = self.evaluate_batch_size
        else:
            self.data_size = self.test_size
            self.batch_size = self.evaluate_batch_size
        
        if split == "train" and self.use_this_many_data > 0:
            self.data = {"answer_string_list": self.dataset[split]["answer_string_list"][: self.use_this_many_data],
                         "sentence_id_list": self.dataset[split]["sentence_id_list"][: self.use_this_many_data],
                         "question_id": self.dataset[split]["question_id"][: self.use_this_many_data],
                         "answer_id_list": self.dataset[split]["answer_id_list"][: self.use_this_many_data],
                         "srl_list": self.dataset[split]["srl_list"][: self.use_this_many_data],
                         "question_srl_list": self.dataset[split]["question_srl_list"][: self.use_this_many_data]}
            self.data_size = self.use_this_many_data
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.current_answers, self.current_answer_id_list = None, None
        self.current_sentence_id_list, self.current_question_id = None, None
        self.current_srl_list, self.current_question_srl_list = None, None
        self.infos = None

    def reset(self, random=True):
        if random is True:
            # randomly sample a batch of d-q-a tuple
            indices = np.random.choice(self.data_size, self.batch_size).tolist()
        else:
            # just take next batch
            if self.split == "train":
                self.batch_pointer = self.train_batch_pointer
            indices = np.arange(self.batch_pointer, self.batch_pointer + self.batch_size).tolist()
            self.batch_pointer += self.batch_size
            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0
            if self.split == "train":
                self.train_batch_pointer = self.batch_pointer
        self.current_answers, self.current_answer_id_list = [], []
        self.current_sentence_id_list, self.current_question_id = [], []
        self.current_srl_list, self.current_question_srl_list = [], []

        for idx in indices:
            if idx >= len(self.data["question_id"]):
                break
            self.current_sentence_id_list.append(self.data["sentence_id_list"][idx])
            self.current_answers.append(self.data["answer_string_list"][idx])
            self.current_question_id.append(self.data["question_id"][idx])
            self.current_answer_id_list.append(self.data["answer_id_list"][idx])
            self.current_srl_list.append(self.data["srl_list"][idx])
            self.current_question_srl_list.append(self.data["question_srl_list"][idx])

        # for each sentence_string_list, randomly sample a sentence as init observation
        obs, srl = [], []
        which_sentence = []
        for _a, _b in zip(self.current_sentence_id_list, self.current_srl_list):
            init_idx = 0 if self.start_from_beginning is True else np.random.choice(len(_a), 1)[0]
            obs.append(_a[init_idx])
            srl.append(_b[init_idx])
            which_sentence.append(init_idx)
        infos = [{"q": q, "a": a, "a_string": a_string, "full_obs": full, "srl": _srl, "full_srl": full_srl, "q_srl": q_srl, "which": which, "stopped": False} for q, a, a_string, full, _srl, full_srl, q_srl, which in zip(self.current_question_id, self.current_answer_id_list, self.current_answers, self.current_sentence_id_list, srl, self.current_srl_list, self.current_question_srl_list, which_sentence)]
        self.infos = infos
        self.last_actions = None
        self.step_counter = 0
        return obs, infos

    def step(self, actions):
        if self.step_counter > self.max_nb_steps_per_episode:
            return None, None
        # given action, return new obs sentence, and update infos
        obs, infos = [], []
        for i in range(len(actions)):
            stopped = False
            if actions[i] == "next":
                new_which = self.infos[i]["which"] + 1
                if new_which >= len(self.current_sentence_id_list[i]):
                    new_which = 0
            elif actions[i] == "previous":
                new_which = self.infos[i]["which"] - 1
                if new_which < 0:
                    new_which = len(self.current_sentence_id_list[i]) - 1
            elif actions[i] == "stop":
                new_which = self.infos[i]["which"]
                stopped = True
            elif actions[i].startswith("ctrl+f"):
                # for now just exact match
                query = actions[i][6:].strip()
                query_id = self.word2id[query]
                curr_which = self.infos[i]["which"]
                for j in range(1, len(self.current_sentence_id_list[i]) + 1):
                    new_which = (curr_which + j) % len(self.current_sentence_id_list[i])
                    sent = self.current_sentence_id_list[i][new_which]
                    if query_id in sent:
                        break
            else:
                raise NotImplementedError
            obs.append(self.current_sentence_id_list[i][new_which])
            infos.append({"q": self.infos[i]["q"], "a": self.infos[i]["a"], "a_string": self.infos[i]["a_string"], "which": new_which, "stopped": stopped, "full_obs": self.infos[i]["full_obs"], "srl": self.current_srl_list[i][new_which], "full_srl": self.infos[i]["full_srl"], "q_srl": self.infos[i]["q_srl"]})

        self.last_actions = actions
        self.infos = infos
        self.step_counter += 1
        return obs, infos

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        np.random.seed(seed)
