import os
import json
import random
import codecs
from tqdm import tqdm
from os.path import join as pjoin

import numpy as np
import gym

EVAL_SIZE = 2000


class ObservationInfomaxData(gym.Env):

    FILENAMES_MAP = {
        "train": "wiki_without_squad_train.json",
        "valid": "wiki_without_squad_validation.json",
        "test": "wiki_without_squad_test.json"
        }

    def __init__(self, config):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)

        # Load dataset splits.
        self.dataset = {}
        for split in ["train", "valid", "test"]:
            self.dataset[split] = {
                "observations": []
                }
            self.load_dataset(split)

        print("loaded dataset from {} ...".format(self.data_path))
        self.train_size = len(self.dataset["train"]["observations"])
        self.valid_size = len(self.dataset["valid"]["observations"])
        self.test_size = len(self.dataset["test"]["observations"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(file_path))
        print(desc)
        data = json.load(codecs.open(file_path, 'r', 'utf-8'))
        assert data["date"] == "20210104"
        data = data["data"]
        for title in tqdm(data, desc=desc):
            self.dataset[split]["observations"].append(data[title])

    def read_config(self):
        self.data_path = self.config["general"]["pretrain_data_path"]
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["training"]["batch_size"]
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
            self.data = {"observations": self.dataset[split]["observations"][: self.use_this_many_data]}
            self.data_size = self.use_this_many_data
        elif split == "train":
            self.data = self.dataset[split]
        else:
            # valid and test, we use 10k data points
            self.data = {"observations": self.dataset[split]["observations"][:EVAL_SIZE]}
            self.data_size = EVAL_SIZE

        self.split = split
        self.batch_pointer = 0

    def get_batch(self):
        if self.split == "train":
            indices = self.rng.choice(self.data_size, self.batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.batch_size

            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0

        positive, negative = [], []
        for idx in indices:
            positive.append(self.data["observations"][idx])
            negative_samples = []        
            negative_indices = self.rng.choice(self.data_size, len(self.data["observations"][idx]), replace=False)

            for j in range(len(self.data["observations"][idx])):
                tmp_idx = self.rng.choice(len(self.data["observations"][negative_indices[j]]))
                negative_samples.append(self.data["observations"][negative_indices[j]][tmp_idx])
            negative.append(negative_samples)

        return positive, negative

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
