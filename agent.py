import copy
import math
import string
import codecs

import numpy as np

import torch
import torch.nn.functional as F
# from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BertModel, BertTokenizer

import memory
from model import LSTM_DQN
from generic import to_np, to_pt, _words_to_ids, pad_sequences, get_qa_reward, get_answer_strings, get_answer_position, get_stopword_ids
from generic import max_len, max_tensor_len, ez_gather_dim_1
from layers import NegativeLogLoss, compute_mask, masked_softmax, masked_mean
from knowledge_graph import DummyGraph, CooccurGraph, RelativePositionGraph, SRLGraph


class ObservationPool(object):

    def __init__(self, capacity=1, max_num_token=-1):
        self.capacity = capacity
        self.max_num_token = max_num_token

    def identical_with_history(self, new_stuff, list_of_old_stuff):
        for i in range(len(list_of_old_stuff)):
            if new_stuff == list_of_old_stuff[i]:
                return True
        return False

    def push_batch(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            self.push_one(i, stuff[i])

    def push_one(self, which, stuff):
        assert which < len(self.memory)
        if not self.identical_with_history(stuff, self.memory[which]):
            self.memory[which].append(stuff)
        if len(self.memory[which]) > self.capacity:
            self.memory[which] = self.memory[which][-self.capacity:]

    def get_last(self):
        return [item[-1] for item in self.memory]

    def get(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            output = []
            for idx in range(len(self.memory[which])):
                output += copy.deepcopy(self.memory[which][idx])
            # use the last max_num_token tokens
            if self.max_num_token > 0:
                output = output[-self.max_num_token:]
            return output

        output = []
        for i in range(len(self.memory)):
            output.append(self.get(which=i))
        return output

    def get_sentence_lists(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            output = copy.deepcopy(self.memory[which])
            return output

        output = []
        for i in range(len(self.memory)):
            output.append(self.get(which=i))
        return output

    def get_sent_list(self):
        return copy.copy(self.memory)

    def reset(self, batch_size):
        self.memory = []
        for _ in range(batch_size):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)


class CustomAgent:
    def __init__(self, config, has_token_set):
        self.mode = "train"
        self.config = config
        self.has_token_set = has_token_set
        print(self.config)
        self.load_config()
        if self.disable_prev_next:
            self.id2action = ["stop", "ctrl+f"]
            self.action2id = {"stop": 0, "ctrl+f": 1}
        else:
            self.id2action = ["previous", "next", "stop", "ctrl+f"]
            self.action2id = {"previous": 0, "next": 1, "stop": 2, "ctrl+f": 3}

        # Load pretrained model/tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        bert_model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        if not self.bert_encoder:
            bert_model.transformer = None
            bert_model.encoder = None
        if self.fine_tune_bert:
            target_bert_model = copy.deepcopy(bert_model)
            bert_model.requires_grad_(True)
            target_bert_model.requires_grad_(False)
        else:
            bert_model.requires_grad_(False)
            target_bert_model = bert_model

        self.word2id = self.bert_tokenizer.get_vocab()
        self.word_vocab = {value:key for key, value in self.word2id.items()}
        self.stopword_ids = get_stopword_ids(self.stopwords, self.bert_tokenizer)
        self.special_token_ids = [self.word2id["[PAD]"], self.word2id["[UNK]"], self.word2id["[CLS]"], self.word2id["[SEP]"], self.word2id["[MASK]"]]

        self.online_net = LSTM_DQN(config=self.config, bert_model=bert_model, word_vocab=self.word_vocab, action_space_size=len(self.id2action))
        self.target_net = LSTM_DQN(config=self.config, bert_model=target_bert_model, word_vocab=self.word_vocab, action_space_size=len(self.id2action))
        self.pretrained_graph_generation_net = LSTM_DQN(config=self.config, bert_model=target_bert_model, word_vocab=self.word_vocab, action_space_size=len(self.id2action))
        self.pretrained_graph_generation_net.eval()
        self.online_net.train()
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False
        for param in self.pretrained_graph_generation_net.parameters():
            param.requires_grad = False

        if self.use_cuda:
            self.online_net.cuda()
            self.target_net.cuda()
            self.pretrained_graph_generation_net.cuda()

        self.naozi = ObservationPool(capacity=self.naozi_capacity, max_num_token=-1)
        if self.enable_graph_input == "false":
            self.kg = DummyGraph()
        elif self.enable_graph_input == "srl":
            self.kg = SRLGraph(tokenizer=self.bert_tokenizer)
        elif self.enable_graph_input == "relative_position":
            self.kg = RelativePositionGraph(stopword_ids=self.stopword_ids, tokenizer=self.bert_tokenizer)
        elif self.enable_graph_input == "cooccur":
            self.kg = CooccurGraph(stopword_ids=self.stopword_ids, tokenizer=self.bert_tokenizer)
        elif self.enable_graph_input == "gata":
            self.kg = DummyGraph()
        else:
            raise NotImplementedError

        self.num_nodes = self.node_capacity
        self.num_relations = self.relation_capacity

        # optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config['training']['optimizer']['learning_rate'])
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']
        # graph input cache
        self.relation_representation_cache = {}

    def load_config(self):
        # stopwords
        self.stopwords = set()
        with codecs.open("./corenlp_stopwords.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                self.stopwords.add(line)
        self.stopwords = self.stopwords | set(string.punctuation)

        self.data_path = self.config['general']['data_path']
        self.qa_reward_prior_threshold = self.config['general']['qa_reward_prior_threshold']
        self.naozi_capacity = self.config['general']['naozi_capacity']
        self.generate_or_point = self.config['general']['generate_or_point']
        self.enable_graph_input = self.config['general']['enable_graph_input']
        self.use_gt_graph = self.config['general']['use_gt_graph']
        self.disable_prev_next = self.config['general']['disable_prev_next']
        self.node_capacity = self.config['general']['node_capacity']
        self.relation_capacity = self.config['general']['relation_capacity']
        self.debug_mode = self.config["general"]["debug_mode"]
        self.fine_tune_bert = self.config["model"]['fine_tune_bert']
        self.bert_encoder = self.config["model"]['bert_encoder']

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.max_episode = self.config['training']['max_episode']
        self.target_net_update_frequency = self.config['training']['target_net_update_frequency']
        self.learn_start_from_this_episode = self.config['training']['learn_start_from_this_episode']
        self.shuffle_sentences_in_qa_training = self.config['training']['shuffle_sentences_in_qa_training']
        self.run_eval = self.config['evaluate']['run_eval']
        self.eval_batch_size = self.config['evaluate']['batch_size']
        self.learning_rate = self.config['training']['optimizer']['learning_rate']
        self.learning_rate_warmup_until = self.config['training']['optimizer']['learning_rate_warmup_until']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.recurrent = self.config['model']['recurrent']

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.report_frequency = self.config['checkpoint']['report_frequency']
        self.load_pretrained = self.config['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['checkpoint']['load_from_tag']
        self.load_parameter_keywords = list(set(self.config['checkpoint']['load_parameter_keywords']))
        self.load_graph_generation_model_from_tag = self.config['checkpoint']['load_graph_generation_model_from_tag']

        self.discount_gamma = self.config['training']['discount_gamma']
        self.qa_loss_lambda = self.config['training']['qa_loss_lambda']
        self.interaction_loss_lambda = self.config['training']['interaction_loss_lambda']
        self.patience = self.config['training']['patience']

        # pre-training
        self.backprop_frequency = self.config['pretraining']['backprop_frequency']

        # replay buffer and updates
        self.replay_batch_size = self.config['replay']['replay_batch_size']
        self.accumulate_reward_from_final = self.config['replay']['accumulate_reward_from_final']

        self.replay_memory = memory.InfoGatheringReplayMemory(self.config['replay']['replay_memory_capacity'],
                                                              priority_fraction=self.config['replay']['replay_memory_priority_fraction'],
                                                              discount_gamma=self.discount_gamma, accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.qa_replay_memory = memory.QAReplayMemory(self.config['replay']['replay_memory_capacity'],
                                                      priority_fraction=self.config['replay']['qa_replay_memory_priority_fraction'])
        self.update_per_k_game_steps = self.config['replay']['update_per_k_game_steps']
        self.multi_step = self.config['replay']['multi_step']
        self.replay_sample_history_length = self.config['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['replay']['replay_sample_update_from']

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def load_pretrained_graph_generation_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading pre-trained graph generation model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            try:
                self.pretrained_graph_generation_net.load_state_dict(pretrained_dict)
                print("Successfully loaded pre-trained graph generation model...")
            except:
                # graph generation net
                model_dict = self.pretrained_graph_generation_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and "action_scorer" not in k)}
                difference_pretrain_this = [k for k, _ in pretrained_dict.items() if k not in model_dict]
                difference_this_pretrain = [k for k, _ in model_dict.items() if k not in pretrained_dict]
                model_dict.update(pretrained_dict)
                self.pretrained_graph_generation_net.load_state_dict(model_dict)
                print("WARNING... Model dict is different with pretrained dict. I'm loading only the parameters with same labels now. Make sure you really want this...")
                print("The loaded parameters are:")
                keys = [key for key in pretrained_dict]
                print(", ".join(keys))
                print("--------------------------")
                print("Parameters in pre-trained model but not in the current model are:")
                print(", ".join(difference_pretrain_this))
                print("--------------------------")
                print("Parameters in current model but not in the pre-trained model are:")
                print(", ".join(difference_this_pretrain))
                print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def load_pretrained_model(self, load_from, load_partial_graph=True):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init(self, obs, infos):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.dones = []
        self.prev_actions = [["" for _ in range(len(obs))]]
        self.prev_step_is_still_interacting = np.ones((len(obs),), dtype="float32")  # 1s and starts to be 0 when previous action is "stop"
        self.naozi.reset(batch_size=len(obs))
        self.kg.reset(node_capacity=self.num_nodes, relation_capacity=self.num_relations, batch_size=len(obs))
        self.not_finished_yet = None

    def get_agent_inputs(self, token_id_list, add_special_token=False):
        if add_special_token:
            input_ids = [[self.word2id["[CLS]"]] + item + [self.word2id["[SEP]"]] for item in token_id_list]
        else:
            input_ids = token_id_list
        input_sentence = to_pt(pad_sequences(input_ids, maxlen=max(max_len(input_ids) + 3, 7)), self.use_cuda)  # 3 --> see layer.DepthwiseSeparableConv.padding
        input_mask = compute_mask(input_sentence)
        return input_sentence, input_mask, input_ids

    def get_game_quest_info(self, infos):
        return [item["q"] for item in infos]
    
    def get_word_mask(self, list_of_query_id_list, list_of_observation_id_list):
        batch_size = len(list_of_query_id_list)
        if self.generate_or_point == "generate":
            word_mask = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
            for _id in self.has_token_set:
                if _id in self.stopword_ids or _id in self.special_token_ids:
                    continue
                word_mask[:, _id] = 1.0
            word_mask = to_pt(word_mask, enable_cuda=self.use_cuda, type="float")
            mask_word_id_list = []
            m = list(self.has_token_set - self.stopword_ids)
            for i in range(batch_size):
                mask_word_id_list.append(m)
            return word_mask, mask_word_id_list

        word_mask_np = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
        mask_word_id_list = []
        for i in range(batch_size):
            mask_word_id_list.append(set())
            for qw_idx in list_of_query_id_list[i]:
                if qw_idx in self.stopword_ids or qw_idx in self.special_token_ids:
                    continue
                word_mask_np[i][qw_idx] = 1.0
                mask_word_id_list[i].add(qw_idx)
            if self.generate_or_point == "qmpoint":
                for ow_idx in list_of_observation_id_list[i]:
                    if ow_idx in self.stopword_ids or ow_idx in self.special_token_ids:
                        continue
                    word_mask_np[i][ow_idx] = 1.0
                    mask_word_id_list[i].add(ow_idx)
        mask_word_id_list = [list(item) for item in mask_word_id_list]
        for i in range(len(mask_word_id_list)):
            if len(mask_word_id_list[i]) == 0:
                mask_word_id_list[i].append(self.word2id[","])  # just in case this list is empty
                word_mask_np[i][self.word2id[","]] = 1.0
                continue
        word_mask = to_pt(word_mask_np, enable_cuda=self.use_cuda, type="float")
        return word_mask, mask_word_id_list

    def generate_commands(self, action_indices, ctrlf_indices):

        action_indices_np = to_np(action_indices)
        ctrlf_indices_np = to_np(ctrlf_indices)
        res_str = []
        batch_size = action_indices_np.shape[0]
        for i in range(batch_size):
            which = action_indices_np[i][0]
            if which == self.action2id["ctrl+f"]:
                which_word = ctrlf_indices_np[i][0]
                res_str.append("ctrl+f " + self.bert_tokenizer.decode([which_word]).strip())
            elif which < len(self.id2action):
                res_str.append(self.id2action[which])
            else:
                raise NotImplementedError
        return res_str

    def choose_random_command(self, action_rank, mask_word_ids=None):
        """
        Generate a command randomly, for epsilon greedy.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(-1)
        if mask_word_ids is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(mask_word_ids[j]))
            indices = np.array(indices)
        action_indices = to_pt(indices, self.use_cuda).unsqueeze(-1)  # batch x 1
        return action_indices

    def choose_maxQ_command(self, action_rank, word_mask=None):
        """
        Generate a command by maximum q values, for epsilon greedy.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if word_mask is not None:
            assert word_mask.size() == action_rank.size(), (word_mask.size().shape, action_rank.size())
            action_rank = action_rank * word_mask
        action_indices = torch.argmax(action_rank, -1, keepdim=True)  # batch x 1
        return action_indices

    def get_model(self, model_name):
        if model_name == "online":
            return self.online_net
        elif model_name == "target":
            return self.target_net
        elif model_name == "graph_generator":
            return self.pretrained_graph_generation_net
        else:
            raise NotImplementedError

    def get_relation_features(self, relation_token_id_list, use_model):
        with torch.no_grad():
            if self.enable_graph_input == "relative_position":
                return self.get_relation_features_relative_position_graph(relation_token_id_list, use_model)
            elif self.enable_graph_input == "cooccur":
                return self.get_relation_features_cooccur_graph(relation_token_id_list, use_model)
            elif self.enable_graph_input == "srl":
                return None
            else:
                return NotImplementedError

    def get_node_features(self, node_token_ids, use_model):
        with torch.no_grad():
            if self.enable_graph_input == "relative_position":
                return self.get_node_features_relative_position_graph(node_token_ids, use_model)
            elif self.enable_graph_input == "cooccur":
                return self.get_node_features_cooccur_graph(node_token_ids, use_model)
            elif self.enable_graph_input == "srl":
                return self.get_node_features_srl_graph(node_token_ids, use_model)
            else:
                return NotImplementedError

    def get_node_features_relative_position_graph(self, node_token_ids, use_model):
        return self.get_node_features_cooccur_graph(node_token_ids, use_model)

    def get_relation_features_relative_position_graph(self, relation_token_id_list, use_model):
        model = self.get_model(use_model)
        relation_representations = torch.zeros(self.num_relations, model.block_hidden_dim)
        if self.use_cuda:
            relation_representations = relation_representations.cuda()
        pos = []
        inputs = []
        for i in range(len(relation_token_id_list[0])):
            if self.mode == "eval" or self.online_net.fine_tune_bert:
                inputs.append(relation_token_id_list[0][i])
                pos.append(i)
            else:
                key = str(relation_token_id_list[0][i])
                if key in self.relation_representation_cache:
                    rep = self.relation_representation_cache[key]
                    if self.use_cuda:
                        rep = rep.cuda()
                    relation_representations[i, :] = rep
                else:
                    inputs.append(relation_token_id_list[0][i])
                    pos.append(i)

        fake_batch_size = 128
        num_batch = (len(inputs) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss = inputs[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words, input_mask, _ = self.get_agent_inputs(ss, add_special_token=False)
            relation_features = model.average_embeddings(input_words, input_mask)  # fake_batch x hid
            rep += torch.unbind(relation_features, 0)  # list of hid

        for r, p, inp in zip(rep, pos, inputs):
            relation_representations[p, :] = r
            if self.mode == "eval" or self.online_net.fine_tune_bert:
                continue
            key = str(inp)
            self.relation_representation_cache[key] = r.clone().cpu()

        relation_representations = relation_representations.unsqueeze(0).repeat(len(relation_token_id_list), 1, 1)
        return relation_representations  # batch x n_relation x hid

    def get_relation_features_cooccur_graph(self, relation_token_id_list, use_model):
        model = self.get_model(use_model)
        relation_representations = torch.zeros(len(relation_token_id_list), self.num_relations, model.block_hidden_dim)
        if self.use_cuda:
            relation_representations = relation_representations.cuda()
        pos = []
        inputs = []
        for b in range(len(relation_token_id_list)):
            for i in range(len(relation_token_id_list[b])):
                if self.mode == "eval" or (not self.online_net.bert_encoder) or self.online_net.fine_tune_bert:
                    inputs.append(relation_token_id_list[b][i])
                    pos.append([b, i])
                else:
                    key = str(relation_token_id_list[b][i])
                    if key in self.relation_representation_cache:
                        rep = self.relation_representation_cache[key]
                        if self.use_cuda:
                            rep = rep.cuda()    
                        relation_representations[b, i, :] = rep
                    else:
                        inputs.append(relation_token_id_list[b][i])
                        pos.append([b, i])

        fake_batch_size = 64
        num_batch = (len(inputs) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss = inputs[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words, input_mask, _ = self.get_agent_inputs(ss)
            relation_features_sequence = model.representation_generator(input_words, input_mask)  # fake_batch x sent_length x hid
            _mask = torch.sum(input_mask, -1)  # fake_batch
            tmp = torch.eq(_mask, 0).float()
            if relation_features_sequence.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            relation_features = torch.sum(relation_features_sequence, 1)  # fake_batch x hid
            relation_features = relation_features / _mask.unsqueeze(-1)

            rep += torch.unbind(relation_features, 0)  # list of hid
        for r, p, inp in zip(rep, pos, inputs):
            relation_representations[p[0], p[1], :] = r
            if self.mode == "eval" or (not self.online_net.bert_encoder) or self.online_net.fine_tune_bert:
                continue
            key = str(inp)
            self.relation_representation_cache[key] = r.clone().cpu()

        return relation_representations  # batch x n_relation x hid

    def get_node_features_cooccur_graph(self, node_token_ids, use_model):
        model = self.get_model(use_model)
        node_representations = torch.zeros(len(node_token_ids), self.num_nodes, model.block_hidden_dim)
        if self.use_cuda:
            node_representations = node_representations.cuda()
        pos = []
        token_ids = []
        for b in range(len(node_token_ids)):
            for i in range(len(node_token_ids[b])):
                token_ids.append([node_token_ids[b][i]])
                pos.append([b, i])
        fake_batch_size = 2048
        num_batch = (len(token_ids) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss_id = token_ids[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words, input_mask, _ = self.get_agent_inputs(ss_id, add_special_token=False)
            node_features = model.average_embeddings(input_words, input_mask)  # fake_batch x hid
            rep += torch.unbind(node_features, 0)  # list of hid
        for r, p in zip(rep, pos):
            node_representations[p[0], p[1], :] = r
        return node_representations  # batch x n_node x hid

    def get_node_features_srl_graph_avg_embedding(self, node_token_ids, use_model):
        model = self.get_model(use_model)
        node_representations = torch.zeros(len(node_token_ids), self.num_nodes, model.block_hidden_dim)
        if self.use_cuda:
            node_representations = node_representations.cuda()
        pos = []
        token_ids = []
        for b in range(len(node_token_ids)):
            for i in range(len(node_token_ids[b])):
                token_ids.append(node_token_ids[b][i])
                pos.append([b, i])
        fake_batch_size = 2048
        num_batch = (len(token_ids) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss_id = token_ids[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words, input_mask, _ = self.get_agent_inputs(ss_id, add_special_token=False)
            node_features = model.average_embeddings(input_words, input_mask)  # fake_batch x hid
            rep += torch.unbind(node_features, 0)  # list of hid
        for r, p in zip(rep, pos):
            node_representations[p[0], p[1], :] = r
        return node_representations  # batch x n_node x hid

    def get_node_features_srl_graph(self, node_token_ids, use_model):
        model = self.get_model(use_model)
        node_representations = torch.zeros(len(node_token_ids), self.num_nodes, model.block_hidden_dim)
        if self.use_cuda:
            node_representations = node_representations.cuda()
        pos = []
        inputs = []
        for b in range(len(node_token_ids)):
            for i in range(len(node_token_ids[b])):
                inputs.append(node_token_ids[b][i])
                pos.append([b, i])

        fake_batch_size = 64
        num_batch = (len(inputs) + fake_batch_size - 1) // fake_batch_size
        rep = []
        for i in range(num_batch):
            ss = inputs[i * fake_batch_size: (i + 1) * fake_batch_size]
            input_words, input_mask, _ = self.get_agent_inputs(ss, add_special_token=False)
            node_features_sequence = model.representation_generator(input_words, input_mask)  # fake_batch x sent_length x hid
            _mask = torch.sum(input_mask, -1)  # fake_batch
            tmp = torch.eq(_mask, 0).float()
            if node_features_sequence.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            node_features = torch.sum(node_features_sequence, 1)  # fake_batch x hid
            node_features = node_features / _mask.unsqueeze(-1)

            rep += torch.unbind(node_features, 0)  # list of hid
        for r, p in zip(rep, pos):
            node_representations[p[0], p[1], :] = r

        return node_representations  # batch x n_node x hid

    def get_gcn_input_features(self, node_vocab, relation_vocab, use_model):
        node_features = self.get_node_features(node_vocab, use_model=use_model)  # batch x n_node x hid
        relation_features = self.get_relation_features(relation_vocab, use_model=use_model)  # batch x n_features x hid
        node_mask = torch.ne(torch.sum(node_features, -1), 0).float()  # batch x n_node
        if self.enable_graph_input == "srl":
            relation_mask = torch.autograd.Variable(torch.ones(node_mask.size(0), self.num_relations))  # batch x n_relation
        else:
            relation_mask = torch.ne(torch.sum(relation_features, -1), 0).float()  # batch x n_relation
        if node_mask.is_cuda:
            relation_mask = relation_mask.cuda()
        return node_features, node_mask, relation_features, relation_mask

    def get_graph_representations(self, node_features, node_mask, relation_features, relation_mask, input_adjacency_matrices, use_model):
        model = self.get_model(use_model)
        # nodes
        node_ids = torch.arange(self.num_nodes).unsqueeze(0).long()  # 1 x num_node
        if self.use_cuda:
            node_ids = node_ids.cuda()
        node_embeddings, _ = model.node_embedding(node_ids)  # 1 x num_node x emb
        node_embeddings = node_embeddings.repeat(node_features.size(0), 1, 1)  # batch x num_node x emb+emb
        node_embeddings = torch.cat([node_features, node_embeddings], dim=-1)  # batch x num_node x emb+emb
        node_embeddings = node_embeddings * node_mask.unsqueeze(-1)

        # relations
        if self.enable_graph_input == "srl":
            relation_ids = torch.arange(self.num_relations).unsqueeze(0).long()  # 1 x num_relation
            if self.use_cuda:
                relation_ids = relation_ids.cuda()
            relation_embeddings, _ = model.relation_embedding(relation_ids)  # 1 x num_node x emb
            relation_embeddings = relation_embeddings.repeat(node_features.size(0), 1, 1)  # batch x num_relation x emb
            rgcn_relation_input = relation_embeddings
        else:
            rgcn_relation_input = relation_features

        # r-gcn
        node_encoding_sequence = model.rgcns(node_embeddings, rgcn_relation_input, input_adjacency_matrices)  # batch x num_node x hid
        node_encoding_sequence = model.graph_representation_prj(node_encoding_sequence)
        node_encoding_sequence = node_encoding_sequence * node_mask.unsqueeze(-1)
        return node_encoding_sequence

    def get_match_representations(self, input_description, input_description_mask, input_quest, input_quest_mask, node_representations, node_mask, node_vocabulary, use_model):
        model = self.get_model(use_model)
        description_representation_sequence = model.representation_generator(input_description, input_description_mask)
        quest_representation_sequence = model.representation_generator(input_quest, input_quest_mask)

        if self.enable_graph_input in ["cooccur", "relative_position"]:
            observable_node_mask = self.kg.get_observable_node_mask(input_description, question_id_matrix=input_quest, node_vocabulary=node_vocabulary)  # batch x num_node
            observable_node_mask = to_pt(observable_node_mask, self.use_cuda, type="float")
            node_mask = node_mask * observable_node_mask

        match_representation_sequence = model.get_match_representations(description_representation_sequence,
                                                                        input_description_mask,
                                                                        quest_representation_sequence,
                                                                        input_quest_mask,
                                                                        node_representations,
                                                                        node_mask)
        match_representation_sequence = match_representation_sequence * input_description_mask.unsqueeze(-1)
        return match_representation_sequence

    def get_ranks(self, input_description, input_description_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, node_vocabulary, previous_dynamics, use_model):
        """
        Given input description tensor, and previous hidden and cell states, call model forward, to get Q values of words.
        """
        model = self.get_model(use_model)
        match_representation_sequence = self.get_match_representations(input_description, input_description_mask, input_quest, input_quest_mask, node_representations, node_mask, node_vocabulary, use_model=use_model)
        if not self.recurrent:
            previous_dynamics = None
        action_rank, ctrlf_rank, current_dynamics = model.action_scorer(match_representation_sequence, input_description_mask, ctrlf_word_mask, previous_dynamics)
        if not self.recurrent:
            current_dynamics = None
        return action_rank, ctrlf_rank, current_dynamics

    def act_greedy(self, obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief):
        with torch.no_grad():
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.deepcopy(obs))
                if not self.use_gt_graph:
                    self.kg.push_batch(copy.deepcopy(obs), previous_commands, [item["srl"] for item in infos])
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.deepcopy(obs[i]))
                        if not self.use_gt_graph:
                            self.kg.push_one(i, copy.deepcopy(obs[i]), previous_commands[i], infos[i]["srl"])

            description_list = self.naozi.get()
            input_description, input_description_mask, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, description_id_list)
            current_belief = [None] * batch_size
            if self.enable_graph_input == "gata":
                current_adjacency_matrix, current_belief, previous_adjacency_matrix = self.graph_update_during_rl(input_description, input_description_mask, previous_belief)
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 0.0:
                        current_adjacency_matrix[i] = previous_adjacency_matrix[i]
                        current_belief[i] = previous_belief[i]
                node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
                node_vocabulary, relation_vocabulary, graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                graph_triplets, node_vocabulary, relation_vocabulary, graph_adj_np = self.kg.get()
                graph_adj = to_pt(graph_adj_np, enable_cuda=self.use_cuda, type="float")
                node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(node_vocabulary, relation_vocabulary, use_model="online")
                node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
            else:
                graph_triplets, node_vocabulary, relation_vocabulary = [None] * batch_size, [None] * batch_size, [None] * batch_size
                node_representations, node_mask, relation_mask = None, None, None

            action_rank, ctrlf_rank, current_dynamics = self.get_ranks(input_description, input_description_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, node_vocabulary, previous_dynamics, use_model="online")  # list of batch x vocab
            action_indices = self.choose_maxQ_command(action_rank)
            ctrlf_indices = self.choose_maxQ_command(ctrlf_rank, ctrlf_word_mask)
            chosen_strings = self.generate_commands(action_indices, ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, action_indices.cpu(), ctrlf_indices.cpu(), node_vocabulary, relation_vocabulary, graph_triplets, to_pt(self.prev_step_is_still_interacting, False, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info, current_dynamics, current_belief

    def act_random(self, obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief):
        with torch.no_grad():
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.deepcopy(obs))
                if not self.use_gt_graph:
                    self.kg.push_batch(copy.deepcopy(obs), previous_commands, [item["srl"] for item in infos])
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.deepcopy(obs[i]))
                        if not self.use_gt_graph:
                            self.kg.push_one(i, copy.deepcopy(obs[i]), previous_commands[i], infos[i]["srl"])
            
            description_list = self.naozi.get()
            input_description, input_description_mask, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, ctrlf_word_ids = self.get_word_mask(quest_id_list, description_id_list)
            current_belief = [None] * batch_size
            if self.enable_graph_input == "gata":
                current_adjacency_matrix, current_belief, previous_adjacency_matrix = self.graph_update_during_rl(input_description, input_description_mask, previous_belief)
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 0.0:
                        current_adjacency_matrix[i] = previous_adjacency_matrix[i]
                        current_belief[i] = previous_belief[i]
                node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
                node_vocabulary, relation_vocabulary, graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                graph_triplets, node_vocabulary, relation_vocabulary, graph_adj_np = self.kg.get()
                graph_adj = to_pt(graph_adj_np, enable_cuda=self.use_cuda, type="float")
                node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(node_vocabulary, relation_vocabulary, use_model="online")
                node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
            else:
                graph_triplets, node_vocabulary, relation_vocabulary = [None] * batch_size, [None] * batch_size, [None] * batch_size
                node_representations, node_mask, relation_mask = None, None, None

            # generate commands for one game step, epsilon greedy is applied, i.e.,
            # there is epsilon of chance to generate random commands
            action_rank, ctrlf_rank, current_dynamics = self.get_ranks(input_description, input_description_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, node_vocabulary, previous_dynamics, use_model="online")  # list of batch x vocab
            action_indices = self.choose_random_command(action_rank)
            ctrlf_indices = self.choose_random_command(ctrlf_rank, ctrlf_word_ids)
            chosen_strings = self.generate_commands(action_indices, ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, action_indices.cpu(), ctrlf_indices.cpu(), node_vocabulary, relation_vocabulary, graph_triplets, to_pt(self.prev_step_is_still_interacting, False, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info, current_dynamics, current_belief

    def act(self, obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief)
            if random:
                return self.act_random(obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief)
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.deepcopy(obs))
                if not self.use_gt_graph:
                    self.kg.push_batch(copy.deepcopy(obs), previous_commands, [item["srl"] for item in infos])
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.deepcopy(obs[i]))
                        if not self.use_gt_graph:
                            self.kg.push_one(i, copy.deepcopy(obs[i]), previous_commands[i], infos[i]["srl"])

            description_list = self.naozi.get()
            input_description, input_description_mask, description_id_list = self.get_agent_inputs(description_list)
            ctrlf_word_mask, ctrlf_word_ids = self.get_word_mask(quest_id_list, description_id_list)
            current_belief = [None] * batch_size
            if self.enable_graph_input == "gata":
                current_adjacency_matrix, current_belief, previous_adjacency_matrix = self.graph_update_during_rl(input_description, input_description_mask, previous_belief)
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 0.0:
                        current_adjacency_matrix[i] = previous_adjacency_matrix[i]
                        current_belief[i] = previous_belief[i]
                node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
                node_vocabulary, relation_vocabulary, graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                graph_triplets, node_vocabulary, relation_vocabulary, graph_adj_np = self.kg.get()
                graph_adj = to_pt(graph_adj_np, enable_cuda=self.use_cuda, type="float")
                node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(node_vocabulary, relation_vocabulary, use_model="online")
                node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
            else:
                graph_triplets, node_vocabulary, relation_vocabulary = [None] * batch_size, [None] * batch_size, [None] * batch_size
                node_representations, node_mask, relation_mask = None, None, None
            # generate commands for one game step, epsilon greedy is applied, i.e.,
            # there is epsilon of chance to generate random commands
            action_rank, ctrlf_rank, current_dynamics = self.get_ranks(input_description, input_description_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, node_vocabulary, previous_dynamics, use_model="online")  # list of batch x vocab
            action_indices_maxq = self.choose_maxQ_command(action_rank)
            action_indices_random = self.choose_random_command(action_rank)
            ctrlf_indices_maxq = self.choose_maxQ_command(ctrlf_rank, ctrlf_word_mask)
            ctrlf_indices_random = self.choose_random_command(ctrlf_rank, ctrlf_word_ids)
            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(input_description.size(0), 1))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon
            less_than_epsilon = to_pt(less_than_epsilon, self.use_cuda, type='long')
            greater_than_epsilon = to_pt(greater_than_epsilon, self.use_cuda, type='long')

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_ctrlf_indices = less_than_epsilon * ctrlf_indices_random + greater_than_epsilon * ctrlf_indices_maxq
            chosen_strings = self.generate_commands(chosen_indices, chosen_ctrlf_indices)

            for i in range(batch_size):
                if chosen_strings[i] == "stop":
                    self.not_finished_yet[i] = 0.0

            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0
            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [description_list, chosen_indices.cpu(), chosen_ctrlf_indices.cpu(), node_vocabulary, relation_vocabulary, graph_triplets, to_pt(self.prev_step_is_still_interacting, False, "float")]

            # cache new info in current game step into caches
            self.prev_actions.append(chosen_strings)
            return chosen_strings, replay_info, current_dynamics, current_belief

    def point_random_position(self, point_distribution, mask):
        """
        Generate a command by random, for epsilon greedy.

        Arguments:
            point_distribution: Q values for each position batch x time x 2.
            mask: position masks.
        """
        batch_size = point_distribution.size(0)
        mask_np = to_np(mask)  # batch x time
        indices = []
        for i in range(batch_size):
            msk = mask_np[i]  # time
            indices.append(np.random.choice(len(msk), 2, p=msk / np.sum(msk, -1)))
        indices = to_pt(np.stack(indices, 0), self.use_cuda)   # batch x 2
        return indices

    def point_maxq_position(self, point_distribution, mask):
        """
        Generate a command by maximum q values, for epsilon greedy.

        Arguments:
            point_distribution: Q values for each position batch x time x 2.
            mask: position masks.
        """
        point_distribution_np = to_np(point_distribution)  # batch x time
        mask_np = to_np(mask)  # batch x time
        point_distribution_np = point_distribution_np - np.min(point_distribution_np) + 1e-2  # minus the min value, so that all values are non-negative
        point_distribution_np = point_distribution_np * np.expand_dims(mask_np, -1)  # batch x time x 2
        indices = np.argmax(point_distribution_np, 1)  # batch x 2
        indices = to_pt(np.array(indices), self.use_cuda)   # batch x 2
        return indices

    def answer_question_act(self, observation_list, quest_list, belief):
        with torch.no_grad():

            batch_size = len(observation_list)
            current_belief = None
            if self.enable_graph_input == "gata":
                current_belief = belief
                graph_adj, node_vocabulary, relation_vocabulary = None, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                _, node_vocabulary, relation_vocabulary, graph_adj_np = self.kg.get()
                graph_adj = to_pt(graph_adj_np, enable_cuda=self.use_cuda, type="float")
            else:
                graph_adj, node_vocabulary, relation_vocabulary = None, [None] * batch_size, [None] * batch_size

            point_rank, mask = self.answer_question(observation_list, quest_list, node_vocabulary, relation_vocabulary, graph_adj, current_belief)  # batch x time x 2
            positions_maxq = self.point_maxq_position(point_rank, mask)  # batch x 2
            return positions_maxq  # batch x 2

    def get_dqn_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None, None

        data = self.replay_memory.get_batch(self.replay_batch_size, self.multi_step)
        if data is None:
            return None, None
        obs_list, quest_list, action_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, rewards, next_obs_list, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, actual_ns = data
        if self.use_cuda:
            action_indices, ctrlf_indices, rewards = action_indices.cuda(), ctrlf_indices.cuda(), rewards.cuda()
            if isinstance(belief, torch.Tensor):
                belief, next_belief = belief.cuda(), next_belief.cuda()
        batch_size = len(obs_list)

        input_observation, input_observation_mask, observation_id_list = self.get_agent_inputs(obs_list)
        input_quest, input_quest_mask, quest_id_list = self.get_agent_inputs(quest_list)
        next_input_observation, next_input_observation_mask, next_observation_id_list = self.get_agent_inputs(next_obs_list)

        if self.enable_graph_input == "gata":
            current_adjacency_matrix = self.hidden_to_adjacency_matrix(belief, batch_size=batch_size, use_model="graph_generator")
            node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
            graph_node_vocabulary, graph_relation_vocabulary, graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
        elif self.enable_graph_input != "false":
            graph_adj = to_pt(self.kg.get_adjacency_matrix(graph_triplets, graph_node_vocabulary, graph_relation_vocabulary), enable_cuda=self.use_cuda, type="float")  # batch x relation x max_n_node x max_n_node
            node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(graph_node_vocabulary, graph_relation_vocabulary, use_model="online")
            node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
        else:
            node_representations, node_mask = None, None

        ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, observation_id_list)
        next_ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, next_observation_id_list)
        action_rank, ctrlf_rank, _ = self.get_ranks(input_observation, input_observation_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, graph_node_vocabulary, None, use_model="online")  # batch x vocab
        # ps_a
        q_value_action = ez_gather_dim_1(action_rank, action_indices).squeeze(1)  # batch
        q_value_ctrlf = ez_gather_dim_1(ctrlf_rank, ctrlf_indices).squeeze(1)  # batch
        is_ctrlf = torch.eq(action_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
        q_value = (q_value_action + q_value_ctrlf * is_ctrlf) / (is_ctrlf + 1)  # masked average
        # q_value = torch.mean(torch.stack([q_value_action, q_value_ctrlf], -1), -1)

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            if self.enable_graph_input == "gata":
                next_adjacency_matrix = self.hidden_to_adjacency_matrix(next_belief, batch_size=batch_size, use_model="graph_generator")
                next_node_representations_online, next_node_mask = self.encode_belief_graph(next_adjacency_matrix, use_model="online")
                next_node_representations_target, next_node_mask = self.encode_belief_graph(next_adjacency_matrix, use_model="target")
                next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                next_graph_adj = to_pt(self.kg.get_adjacency_matrix(next_graph_triplets, next_graph_node_vocabulary, next_graph_relation_vocabulary), enable_cuda=self.use_cuda, type="float")  # batch x relation x max_n_node x max_n_node
                # online
                next_node_features, next_node_mask, next_relation_features, next_relation_mask = self.get_gcn_input_features(next_graph_node_vocabulary, next_graph_relation_vocabulary, use_model="online")
                next_node_representations_online = self.get_graph_representations(next_node_features, next_node_mask, next_relation_features, next_relation_mask, next_graph_adj, use_model="online")  # batch x max_n_node x hid
                # target
                next_node_features, next_node_mask, next_relation_features, next_relation_mask = self.get_gcn_input_features(next_graph_node_vocabulary, next_graph_relation_vocabulary, use_model="target")
                next_node_representations_target = self.get_graph_representations(next_node_features, next_node_mask, next_relation_features, next_relation_mask, next_graph_adj, use_model="target")  # batch x max_n_node x hid
            else:
                next_node_representations_online, next_node_representations_target, next_node_mask = None, None, None

            # pns Probabilities p(s_t+n, ·; θonline)
            next_action_rank, next_ctrlf_rank, _ = self.get_ranks(next_input_observation, next_input_observation_mask, input_quest, input_quest_mask, next_ctrlf_word_mask, next_node_representations_online, next_node_mask, next_graph_node_vocabulary, None, use_model="online")  # batch x vocab
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_command(next_action_rank)  # batch x 1
            next_ctrlf_indices = self.choose_maxQ_command(next_ctrlf_rank, next_ctrlf_word_mask)  # batch x 1
            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_action_rank, next_ctrlf_rank, _ = self.get_ranks(next_input_observation, next_input_observation_mask, input_quest, input_quest_mask, next_ctrlf_word_mask, next_node_representations_target, next_node_mask, next_graph_node_vocabulary, None, use_model="target")  # batch x vocab
            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value_action = ez_gather_dim_1(next_action_rank, next_action_indices).squeeze(1)  # batch
            next_q_value_ctrlf = ez_gather_dim_1(next_ctrlf_rank, next_ctrlf_indices).squeeze(1)  # batch
            next_is_ctrlf = torch.eq(next_action_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
            next_q_value = (next_q_value_action + next_q_value_ctrlf * next_is_ctrlf) / (next_is_ctrlf + 1)  # masked average

            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value
        
    def get_drqn_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.replay_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.replay_sample_history_length)
        if data is None:
            return None, None
        seq_obs, quest, seq_word_indices, seq_ctrlf_indices, seq_graph_node_vocabulary, seq_graph_relation_vocabulary, seq_graph_triplets, seq_belief, seq_reward, seq_next_obs, seq_next_graph_node_vocabulary, seq_next_graph_relation_vocabulary, seq_next_graph_triplets, seq_next_belief, seq_trajectory_mask = data
        sum_loss, sum_q_value, none_zero = None, None, None
        prev_dynamics = None
        input_quest, input_quest_mask, quest_id_list = self.get_agent_inputs(quest)
        batch_size = len(input_quest)

        for step_no in range(self.replay_sample_history_length):
            obs, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward, next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, trajectory_mask = seq_obs[step_no], seq_word_indices[step_no], seq_ctrlf_indices[step_no], seq_graph_node_vocabulary[step_no], seq_graph_relation_vocabulary[step_no], seq_graph_triplets[step_no], seq_belief[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_graph_node_vocabulary[step_no], seq_next_graph_relation_vocabulary[step_no], seq_next_graph_triplets[step_no], seq_next_belief[step_no], seq_trajectory_mask[step_no]
            if self.use_cuda:
                word_indices, ctrlf_indices, reward, trajectory_mask = word_indices.cuda(), ctrlf_indices.cuda(), reward.cuda(), trajectory_mask.cuda()
                if isinstance(belief, torch.Tensor):
                    belief, next_belief = belief.cuda(), next_belief.cuda()

            input_observation, input_observation_mask, observation_id_list = self.get_agent_inputs(obs)
            next_input_observation, next_input_observation_mask, next_observation_id_list = self.get_agent_inputs(next_obs)

            if self.enable_graph_input == "gata":
                current_adjacency_matrix = self.hidden_to_adjacency_matrix(belief, batch_size=batch_size, use_model="graph_generator")
                node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
                graph_node_vocabulary, graph_relation_vocabulary, graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
            elif self.enable_graph_input != "false":
                graph_adj = to_pt(self.kg.get_adjacency_matrix(graph_triplets, graph_node_vocabulary, graph_relation_vocabulary), enable_cuda=self.use_cuda, type="float")  # batch x relation x max_n_node x max_n_node
                node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(graph_node_vocabulary, graph_relation_vocabulary, use_model="online")
                node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
            else:
                node_representations, node_mask = None, None
            ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, observation_id_list)
            next_ctrlf_word_mask, _ = self.get_word_mask(quest_id_list, next_observation_id_list)

            action_rank, ctrlf_rank, current_dynamics = self.get_ranks(input_observation, input_observation_mask, input_quest, input_quest_mask, ctrlf_word_mask, node_representations, node_mask, graph_node_vocabulary, prev_dynamics, use_model="online")  # batch x vocab
            # ps_a
            q_value_action = ez_gather_dim_1(action_rank, word_indices).squeeze(1)  # batch
            q_value_ctrlf = ez_gather_dim_1(ctrlf_rank, ctrlf_indices).squeeze(1)  # batch
            is_ctrlf = torch.eq(word_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
            q_value = (q_value_action + q_value_ctrlf * is_ctrlf) / (is_ctrlf + 1)  # masked average
            # q_value = torch.mean(torch.stack([q_value_action, q_value_ctrlf], -1), -1)

            prev_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.replay_sample_update_from:
                q_value = q_value.detach()
                prev_dynamics = prev_dynamics.detach()
                continue
            
            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)

                if self.enable_graph_input == "gata":
                    next_adjacency_matrix = self.hidden_to_adjacency_matrix(next_belief, batch_size=batch_size, use_model="graph_generator")
                    next_node_representations_online, next_node_mask = self.encode_belief_graph(next_adjacency_matrix, use_model="online")
                    next_node_representations_target, next_node_mask = self.encode_belief_graph(next_adjacency_matrix, use_model="target")
                    next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets = [None] * batch_size, [None] * batch_size, [None] * batch_size
                elif self.enable_graph_input != "false":
                    next_graph_adj = to_pt(self.kg.get_adjacency_matrix(next_graph_triplets, next_graph_node_vocabulary, next_graph_relation_vocabulary), enable_cuda=self.use_cuda, type="float")  # batch x relation x max_n_node x max_n_node
                    # online
                    next_node_features, next_node_mask, next_relation_features, next_relation_mask = self.get_gcn_input_features(next_graph_node_vocabulary, next_graph_relation_vocabulary, use_model="online")
                    next_node_representations_online = self.get_graph_representations(next_node_features, next_node_mask, next_relation_features, next_relation_mask, next_graph_adj, use_model="online")  # batch x max_n_node x hid
                    # target
                    next_node_features, next_node_mask, next_relation_features, next_relation_mask = self.get_gcn_input_features(next_graph_node_vocabulary, next_graph_relation_vocabulary, use_model="target")
                    next_node_representations_target = self.get_graph_representations(next_node_features, next_node_mask, next_relation_features, next_relation_mask, next_graph_adj, use_model="target")  # batch x max_n_node x hid
                else:
                    next_node_representations_online, next_node_representations_target, next_node_mask = None, None, None

                # pns Probabilities p(s_t+n, ·; θonline)
                next_action_rank, next_ctrlf_rank, _ = self.get_ranks(next_input_observation, next_input_observation_mask, input_quest, input_quest_mask, next_ctrlf_word_mask, next_node_representations_online, next_node_mask, next_graph_node_vocabulary, prev_dynamics, use_model="online")  # batch x vocab
                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_command(next_action_rank)  # batch x 1
                next_ctrlf_indices = self.choose_maxQ_command(next_ctrlf_rank, next_ctrlf_word_mask)  # batch x 1
                # pns # Probabilities p(s_t+n, ·; θtarget)
                next_action_rank, next_ctrlf_rank, _ = self.get_ranks(next_input_observation, next_input_observation_mask, input_quest, input_quest_mask, next_ctrlf_word_mask, next_node_representations_target, next_node_mask, next_graph_node_vocabulary, prev_dynamics, use_model="target")  # batch x vocab
                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value_action = ez_gather_dim_1(next_action_rank, next_action_indices).squeeze(1)  # batch
                next_q_value_ctrlf = ez_gather_dim_1(next_ctrlf_rank, next_ctrlf_indices).squeeze(1)  # batch
                next_is_ctrlf = torch.eq(next_action_indices, float(self.action2id["ctrl+f"])).float()  # when the action is ctrl+f, batch
                next_q_value = (next_q_value_action + next_q_value_ctrlf * next_is_ctrlf) / (next_is_ctrlf + 1)  # masked average

            reward = reward + next_q_value * self.discount_gamma  # batch
            loss = F.smooth_l1_loss(q_value, reward, reduction="none")  # batch
            loss = loss * trajectory_mask  # batch
            if sum_loss is None:
                sum_loss = torch.sum(loss)
                sum_q_value = torch.sum(q_value)
                none_zero = torch.sum(trajectory_mask)
            else:
                sum_loss = sum_loss + torch.sum(loss)
                none_zero = none_zero + torch.sum(trajectory_mask)
                sum_q_value = sum_q_value + torch.sum(q_value)

        tmp = torch.eq(none_zero, 0).float()  # 1
        if sum_loss.is_cuda:
            tmp = tmp.cuda()
        none_zero = none_zero + tmp  # 1
        loss = sum_loss / none_zero
        q_value = sum_q_value / none_zero
        return loss, q_value

    def update_interaction(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            interaction_loss, q_value = self.get_drqn_loss()
        else:
            interaction_loss, q_value = self.get_dqn_loss()
        if interaction_loss is None:
            return None, None
        loss = interaction_loss * self.interaction_loss_lambda
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(interaction_loss)), to_np(torch.mean(q_value))

    def get_qa_loss(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.qa_replay_memory) < self.replay_batch_size:
            return None
        transitions = self.qa_replay_memory.sample(self.replay_batch_size)
        batch = memory.qa_transition(*zip(*transitions))

        batch_observation = []
        for b in range(len(batch.observation_list)):
            indices = np.arange(len(batch.observation_list[b]))
            if self.shuffle_sentences_in_qa_training:
                np.random.shuffle(indices)
            obs = []
            for idx in indices:
                obs += batch.observation_list[b][idx]
            batch_observation.append(obs)

        answer_token_ids = [item[0] for item in batch.answer_token_ids]
        groundtruth_answer_positions = get_answer_position(batch_observation, answer_token_ids)  # list: batch x 2
        batch_observation = [o for pos, o in zip(groundtruth_answer_positions, batch_observation) if pos is not None]
        quest_list = [q for pos, q in zip(groundtruth_answer_positions, batch.quest_list) if pos is not None]
        graph_triplets = [gt for pos, gt in zip(groundtruth_answer_positions, batch.graph_triplets) if pos is not None]
        graph_node_vocabulary = [v for pos, v in zip(groundtruth_answer_positions, batch.graph_node_vocabulary) if pos is not None]
        graph_relation_vocabulary = [v for pos, v in zip(groundtruth_answer_positions, batch.graph_relation_vocabulary) if pos is not None]
        belief = [item for pos, item in zip(groundtruth_answer_positions, batch.belief) if pos is not None]
        groundtruth_answer_positions = [item for item in groundtruth_answer_positions if item is not None]  # list: batch x 2
        if len(groundtruth_answer_positions) == 0:
            return None

        if isinstance(belief[0], torch.Tensor):
            belief = torch.stack(belief, 0)  # batch x hid
            if self.use_cuda:
                belief = belief.cuda()

        if self.enable_graph_input == "gata":
            graph_node_vocabulary, graph_relation_vocabulary, graph_adj = None, None, None
        elif self.enable_graph_input != "false":
            graph_adj = to_pt(self.kg.get_adjacency_matrix(graph_triplets, graph_node_vocabulary, graph_relation_vocabulary), enable_cuda=self.use_cuda, type="float")  # batch x max_n_node x max_n_node
        else:
            graph_node_vocabulary, graph_relation_vocabulary, graph_adj = None, None, None

        answer_distribution, obs_mask = self.answer_question(batch_observation, quest_list, graph_node_vocabulary, graph_relation_vocabulary, graph_adj, belief)  # answer_distribution is batch x time x 2
        answer_distribution = masked_softmax(answer_distribution, obs_mask.unsqueeze(-1), axis=1)

        groundtruth = pad_sequences(groundtruth_answer_positions).astype('int32')
        groundtruth = to_pt(groundtruth, self.use_cuda)  # batch x 2
        batch_loss = NegativeLogLoss(answer_distribution * obs_mask.unsqueeze(-1), groundtruth)

        return torch.mean(batch_loss)

    def update_qa(self):
        # update neural model by replaying snapshots in replay memory
        qa_loss = self.get_qa_loss()
        if qa_loss is None:
            return None
        loss = qa_loss * self.qa_loss_lambda
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(qa_loss))

    def answer_question(self, observation_list, quest_list, graph_node_vocabulary, graph_relation_vocabulary, graph_adj, current_belief):
        # first pad matching_representation_sequence, and get the mask
        model = self.get_model("online")

        input_observation, input_observation_mask, _ = self.get_agent_inputs(observation_list)
        input_quest, input_quest_mask, _ = self.get_agent_inputs(quest_list)
        batch_size = len(input_observation)
        if self.enable_graph_input == "gata":
            current_adjacency_matrix = self.hidden_to_adjacency_matrix(current_belief, batch_size=batch_size, use_model="graph_generator")
            node_representations, node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
            graph_node_vocabulary = [None] * batch_size
        elif self.enable_graph_input != "false":
            node_features, node_mask, relation_features, relation_mask = self.get_gcn_input_features(graph_node_vocabulary, graph_relation_vocabulary, use_model="online")
            node_representations = self.get_graph_representations(node_features, node_mask, relation_features, relation_mask, graph_adj, use_model="online")  # batch x max_n_node x hid
        else:
            node_representations, node_mask = None, None

        matching_representation_sequence = self.get_match_representations(input_observation, input_observation_mask, input_quest, input_quest_mask, node_representations, node_mask, graph_node_vocabulary, use_model="online")
        # get mask
        mask = compute_mask(input_observation)
        # returns batch x time x 2
        point_rank = model.answer_question(matching_representation_sequence, mask)

        return point_rank, mask

    def finish_of_episode(self, episode_no, batch_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()

        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
            self.epsilon = max(self.epsilon, 0.0)

        # learning rate warmup
        if batch_no < self.learning_rate_warmup_until:
            cr = self.learning_rate / math.log2(self.learning_rate_warmup_until)
            learning_rate = cr * math.log2(batch_no + 1)
        else:
            learning_rate = self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    # Deep Infomax Specific

    def encode_belief_graph(self, input_adjacency_matrices, use_model):
        model = self.get_model(use_model)
        batch_size = input_adjacency_matrices.size(0)
        # nodes
        node_ids = torch.arange(self.num_nodes).unsqueeze(0).long()  # 1 x num_node
        if self.use_cuda:
            node_ids = node_ids.cuda()
        node_embeddings, _ = model.node_embedding(node_ids)  # 1 x num_node x emb
        node_embeddings = node_embeddings.repeat(batch_size, 1, 1)  # batch x num_node x emb

        # relations
        relation_ids = torch.arange(self.num_relations).unsqueeze(0).long()  # 1 x num_relation
        if self.use_cuda:
            relation_ids = relation_ids.cuda()
        relation_embeddings, _ = model.relation_embedding(relation_ids)  # 1 x num_node x emb
        relation_embeddings = relation_embeddings.repeat(batch_size, 1, 1)  # batch x num_relation x emb

        # r-gcn
        node_encoding_sequence = model.rgcns(node_embeddings, relation_embeddings, input_adjacency_matrices)  # batch x num_node x hid
        node_encoding_sequence = model.graph_representation_prj(node_encoding_sequence)
        node_mask = torch.ones(batch_size, self.num_nodes)
        if self.use_cuda:
            node_mask = node_mask.cuda()
        return node_encoding_sequence, node_mask

    def hidden_to_adjacency_matrix(self, hidden, batch_size, use_model):
        model = self.get_model(use_model)
        if hidden is None:
            adjacency_matrix = torch.zeros(batch_size, self.num_relations, self.num_nodes, self.num_nodes)
            if self.use_cuda:
                adjacency_matrix = adjacency_matrix.cuda()
        else:
            adjacency_matrix = model.belief_linear_1(hidden)
            adjacency_matrix = torch.relu(adjacency_matrix)
            adjacency_matrix = model.belief_linear_2(adjacency_matrix)
            adjacency_matrix = torch.tanh(adjacency_matrix)
            adjacency_matrix = adjacency_matrix.view(batch_size, int(self.num_relations / 2), self.num_nodes, self.num_nodes)
            adjacency_matrix = adjacency_matrix.repeat(1, 2, 1, 1)
            for i in range(int(self.num_relations / 2)):
                adjacency_matrix[:, int(self.num_relations / 2) + i] = adjacency_matrix[:, i].permute(0, 2, 1)
        return adjacency_matrix

    def graph_update_during_rl(self, input_description, input_description_mask, h_t_minus_one):
        # input_description: batch x sent_len
        # input_description_mask: batch x sent_len
        # h_t_minus_one: batch x hid
        batch_size = len(input_description)
        # encode
        description_representation_sequence = self.pretrained_graph_generation_net.representation_generator(input_description, input_description_mask)
        # decode graph from history
        prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=batch_size, use_model="graph_generator")
        # encode history graph
        node_encoding_sequence, node_mask = self.encode_belief_graph(prev_adjacency_matrix, use_model="graph_generator")
        # co-attention
        h_og = self.pretrained_graph_generation_net.belief_attention(description_representation_sequence, node_encoding_sequence, input_description_mask, node_mask)
        h_go = self.pretrained_graph_generation_net.belief_attention(node_encoding_sequence, description_representation_sequence, node_mask, input_description_mask)
        h_og = self.pretrained_graph_generation_net.belief_attention_prj(h_og) # batch X len X hid
        h_go = self.pretrained_graph_generation_net.belief_attention_prj(h_go) # batch X len X hid
        ave_h_go = masked_mean(h_go, m=node_mask, dim=1)  # batch x hid
        ave_h_og = masked_mean(h_og, m=input_description_mask, dim=1)  # batch x hid

        rnn_input = self.pretrained_graph_generation_net.belief_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og], dim=1))  # batch x hid
        rnn_input = torch.tanh(rnn_input)  # batch x hid
        h_t = self.pretrained_graph_generation_net.belief_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.pretrained_graph_generation_net.belief_graph_rnncell(rnn_input)  # both batch x hid
        current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=batch_size, use_model="graph_generator")
        return current_adjacency_matrix, h_t, prev_adjacency_matrix

    def get_observation_infomax_logits(self, batch_positive, batch_negative, h_t_minus_one, prev_adjacency_matrix=None, prev_node_encoding_sequence=None, prev_node_mask=None):
        # batch positive: batch x sent_len
        # batch negative: batch x sent_len

        # encode
        input_positive, positive_mask, _ = self.get_agent_inputs(batch_positive)
        input_negative, negative_mask, _ = self.get_agent_inputs(batch_negative)
        positive_representation_sequence = self.online_net.representation_generator(input_positive, positive_mask)
        negative_representation_sequence = self.online_net.representation_generator(input_negative, negative_mask)

        if prev_node_encoding_sequence is None:
            if prev_adjacency_matrix is None:
                # decode graph from history
                prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=len(batch_positive), use_model="online")
            # encode history graph
            prev_node_encoding_sequence, prev_node_mask = self.encode_belief_graph(prev_adjacency_matrix, use_model="online")

        # co-attention
        h_og = self.online_net.belief_attention(positive_representation_sequence, prev_node_encoding_sequence, positive_mask, prev_node_mask)
        h_go = self.online_net.belief_attention(prev_node_encoding_sequence, positive_representation_sequence, prev_node_mask, positive_mask)
        h_og = self.online_net.belief_attention_prj(h_og) # batch X len X hid
        h_go = self.online_net.belief_attention_prj(h_go) # batch X len X hid
        ave_h_go = masked_mean(h_go, m=prev_node_mask, dim=1)  # batch x hid
        ave_h_og = masked_mean(h_og, m=positive_mask, dim=1)  # batch x hid

        rnn_input = self.online_net.belief_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og], dim=1))  # batch x hid
        rnn_input = torch.tanh(rnn_input)  # batch x hid
        h_t = self.online_net.belief_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.online_net.belief_graph_rnncell(rnn_input)  # both batch x hid
        current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=len(batch_positive), use_model="online")
        current_node_encoding_sequence, current_node_mask = self.encode_belief_graph(current_adjacency_matrix, use_model="online")
        new_belief_graph_representations = masked_mean(current_node_encoding_sequence, m=current_node_mask, dim=1)  # batch x hid

        logits = self.online_net.observation_discriminator(new_belief_graph_representations, positive_representation_sequence, positive_mask, negative_representation_sequence, negative_mask) 
        return logits, h_t, current_adjacency_matrix, current_node_encoding_sequence, current_node_mask

    def get_observation_infomax_loss(self, positive_samples, negative_samples, evaluate=False):
        # positive_samples: batch x num_sent x len_sent
        # negative_samples: batch x num_sent x len_sent

        batch_size = len(positive_samples)
        num_sentences = [len(elem) for elem in positive_samples]
        max_num_sentences = max(num_sentences)
        episodes_masks = torch.zeros((batch_size, max_num_sentences), dtype=torch.float)
        if self.use_cuda:
            episodes_masks = episodes_masks.cuda()

        for i in range(batch_size):
            episodes_masks[i, :num_sentences[i]] = 1.0
        episodes_masks = episodes_masks.repeat(2, 1)  # batch*2 x max_num_sent, repeat for corrupted obs 
        previous_graph_hidden_state = None
        prev_adjacency_matrix, prev_node_encoding_sequence, prev_node_mask = None, None, None

        last_k_batches_loss = []
        return_losses = []
        return_accuracies = []
        
        for i in range(max_num_sentences):
            current_step_eps_masks = episodes_masks[:, i]
            batch_positive, batch_negative = [], []
            for j in range(len(positive_samples)):
                if i >= len(positive_samples[j]):
                    batch_positive.append([])
                    batch_negative.append([])
                else:
                    batch_positive.append(positive_samples[j][i])
                    batch_negative.append(negative_samples[j][i])

            logits, current_graph_hidden_state, current_adjacency_matrix, current_node_encoding_sequence, current_node_mask = self.get_observation_infomax_logits(batch_positive, batch_negative, previous_graph_hidden_state, prev_adjacency_matrix, prev_node_encoding_sequence, prev_node_mask)
            previous_graph_hidden_state = current_graph_hidden_state
            prev_adjacency_matrix, prev_node_encoding_sequence, prev_node_mask = current_adjacency_matrix, current_node_encoding_sequence, current_node_mask

            # labels
            labels_positive = torch.ones(batch_size) # batch
            labels_negative = torch.zeros(batch_size) # batch
            labels = torch.cat([labels_positive, labels_negative]) # batch*2
            if self.use_cuda:
                labels = labels.cuda()

            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits.squeeze(1), labels)
            loss = torch.sum(loss * current_step_eps_masks) / torch.sum(current_step_eps_masks)
            return_losses.append(to_np(loss))
            preds = to_np(logits.squeeze(1))
            preds = (preds > 0.5).astype("int32")
            for m in range(batch_size):
                if current_step_eps_masks[m] == 0:
                    continue
                return_accuracies.append(float(preds[m] == 1))
            for m in range(batch_size):
                if current_step_eps_masks[m] == 0:
                    continue
                return_accuracies.append(float(preds[batch_size + m] == 0))
            if evaluate:
                continue
            last_k_batches_loss.append(loss.unsqueeze(0))
            if ((i + 1) % self.backprop_frequency == 0 or i == max_num_sentences - 1) and i > 0:
                self.optimizer.zero_grad()
                torch_last_k_batches_loss = torch.cat(last_k_batches_loss)
                ave_k_loss = torch.mean(torch_last_k_batches_loss)
                ave_k_loss.backward()
                self.optimizer.step()
                last_k_batches_loss = []
                previous_graph_hidden_state = previous_graph_hidden_state.detach()
                prev_adjacency_matrix = prev_adjacency_matrix.detach()
                prev_node_encoding_sequence = prev_node_encoding_sequence.detach()
                prev_node_mask = prev_node_mask.detach()

        return return_losses, return_accuracies
