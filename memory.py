import copy
from collections import namedtuple
import numpy as np
import torch
from generic import to_pt


# a snapshot of state to be stored in replay memory
info_gathering_transition = namedtuple('transition', ('observation_list', 'quest_list', 'word_indices', 'ctrlf_indices', 'graph_node_vocabulary', 'graph_relation_vocabulary', 'graph_triplets', 'belief', 'reward'))
# a snapshot of state to be stored in replay memory for question answering
qa_transition = namedtuple('qa_transition', ('observation_list', 'quest_list', 'graph_node_vocabulary', 'graph_relation_vocabulary', 'graph_triplets', 'answer_token_ids', 'belief'))


class InfoGatheringReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma=1.0, accumulate_reward_from_final=False, seed=None):
        # prioritized replay memory
        self.rng = np.random.RandomState(seed)
        self.priority_fraction = priority_fraction
        self.discount_gamma = discount_gamma
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.accumulate_reward_from_final = accumulate_reward_from_final

    def push(self, is_prior, t):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        trajectory = []
        for i in range(len(t)):
            trajectory.append(info_gathering_transition(t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5], t[i][6], t[i][7], t[i][8]))
        if is_prior:
            self.alpha_memory.append(trajectory)
            if len(self.alpha_memory) > self.alpha_capacity:
                remove_id = self.rng.randint(self.alpha_capacity)
                self.alpha_memory = self.alpha_memory[:remove_id] + self.alpha_memory[remove_id + 1:]
        else:
            self.beta_memory.append(trajectory)
            if len(self.beta_memory) > self.beta_capacity:
                remove_id = self.rng.randint(self.beta_capacity)
                self.beta_memory = self.beta_memory[:remove_id] + self.beta_memory[remove_id + 1:]

    def _get_single_transition(self, n, which_memory):
        if len(which_memory) == 0:
            return None
        assert n > 0
        trajectory_id = self.rng.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        if len(trajectory) <= n:
            return None
        head = self.rng.randint(0, len(trajectory) - n)
        final = len(trajectory) - 1

        # all good
        obs = trajectory[head].observation_list
        quest = trajectory[head].quest_list
        word_indices = trajectory[head].word_indices
        ctrlf_indices = trajectory[head].ctrlf_indices
        graph_node_vocabulary = trajectory[head].graph_node_vocabulary
        graph_relation_vocabulary = trajectory[head].graph_relation_vocabulary
        graph_triplets = trajectory[head].graph_triplets
        belief = trajectory[head].belief
        next_obs = trajectory[head + n].observation_list
        next_graph_node_vocabulary = trajectory[head + n].graph_node_vocabulary
        next_graph_relation_vocabulary = trajectory[head + n].graph_relation_vocabulary
        next_graph_triplets = trajectory[head + n].graph_triplets
        next_belief = trajectory[head + n].belief

        # 1 2 [3] 4 5 (6) 7 8 9f
        how_long = final - head + 1 if self.accumulate_reward_from_final else n + 1
        accumulated_rewards = [self.discount_gamma ** i * trajectory[head + i].reward for i in range(how_long)]
        accumulated_rewards = accumulated_rewards[:n + 1]
        reward = torch.sum(torch.stack(accumulated_rewards))

        return (obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward, next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, n)

    def _get_batch(self, n_list, which_memory):
        res = []
        for i in range(len(n_list)):
            output = self._get_single_transition(n_list[i], which_memory)
            if output is None:
                continue
            res.append(output)
        if len(res) == 0:
            return None
        return res

    def get_batch(self, batch_size, multi_step=1):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))
        res = []
        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch(self.rng.randint(1, multi_step + 1, size=from_alpha), self.alpha_memory)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch(self.rng.randint(1, multi_step + 1, size=from_beta), self.beta_memory)
        if res_alpha is None and res_beta is None:
            return None
        if res_alpha is not None:
            res += res_alpha
        if res_beta is not None:
            res += res_beta
        self.rng.shuffle(res)

        obs_list, quest_list, word_indices_list, ctrlf_indices_list, graph_node_vocabulary_list, graph_relation_vocabulary_list, graph_triplet_list, belief_list, reward_list, actual_n_list = [], [], [], [], [], [], [], [], [], []
        next_obs_list, next_graph_node_vocabulary_list, next_graph_relation_vocabulary_list, next_graph_triplet_list, next_belief_list = [], [], [], [], []

        for item in res:
            obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward, next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, n = item
            obs_list.append(obs)
            quest_list.append(quest)
            word_indices_list.append(word_indices)
            ctrlf_indices_list.append(ctrlf_indices)
            graph_node_vocabulary_list.append(graph_node_vocabulary)
            graph_relation_vocabulary_list.append(graph_relation_vocabulary)
            graph_triplet_list.append(graph_triplets)
            belief_list.append(belief)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            next_graph_node_vocabulary_list.append(next_graph_node_vocabulary)
            next_graph_relation_vocabulary_list.append(next_graph_relation_vocabulary)
            next_graph_triplet_list.append(next_graph_triplets)
            next_belief_list.append(next_belief)
            actual_n_list.append(n)

        word_indices_list = torch.stack(word_indices_list, 0)  # batch x 1
        ctrlf_indices_list = torch.stack(ctrlf_indices_list, 0)  # batch x 1
        reward_list = torch.stack(reward_list, 0)  # batch
        actual_n_list = np.array(actual_n_list)
        if isinstance(belief_list[0], torch.Tensor):
            belief_list = torch.stack(belief_list, 0)  # batch x hid
            next_belief_list = torch.stack(next_belief_list, 0)  # batch x hid

        return [obs_list, quest_list, word_indices_list, ctrlf_indices_list, graph_node_vocabulary_list, graph_relation_vocabulary_list, graph_triplet_list, belief_list, reward_list, next_obs_list, next_graph_node_vocabulary_list, next_graph_relation_vocabulary_list, next_graph_triplet_list, next_belief_list, actual_n_list]

    def _get_single_sequence_transition(self, which_memory, sample_history_length, contains_first_step):
        if len(which_memory) == 0:
            return None
        assert sample_history_length > 1
        trajectory_id = self.rng.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        _padded_trajectory = copy.deepcopy(trajectory)
        trajectory_mask = [1.0 for _ in range(len(_padded_trajectory))]
        if contains_first_step:
            while len(_padded_trajectory) <= sample_history_length:
                _padded_trajectory = _padded_trajectory + [copy.copy(_padded_trajectory[-1])]
                trajectory_mask.append(0.0)
            head = 0
        else:
            if len(_padded_trajectory) - sample_history_length <= 1:
                return None
            head = self.rng.randint(1, len(_padded_trajectory) - sample_history_length)
        # tail = head + sample_history_length - 1
        final = len(_padded_trajectory) - 1

        seq_obs, seq_word_indices, seq_ctrlf_indices, seq_graph_node_vocabulary, seq_graph_relation_vocabulary, seq_graph_triplets, seq_belief, seq_reward = [], [], [], [], [], [], [], []
        seq_next_obs, seq_next_graph_node_vocabulary, seq_next_graph_relation_vocabulary, seq_next_graph_triplets, seq_next_belief = [], [], [], [], []
        quest = _padded_trajectory[head].quest_list
        for j in range(sample_history_length):
            seq_obs.append(_padded_trajectory[head + j].observation_list)
            seq_word_indices.append(_padded_trajectory[head + j].word_indices)
            seq_ctrlf_indices.append(_padded_trajectory[head + j].ctrlf_indices)
            seq_graph_node_vocabulary.append(_padded_trajectory[head + j].graph_node_vocabulary)
            seq_graph_relation_vocabulary.append(_padded_trajectory[head + j].graph_relation_vocabulary)
            seq_graph_triplets.append(_padded_trajectory[head + j].graph_triplets)
            seq_belief.append(_padded_trajectory[head + j].belief)
            seq_next_obs.append(_padded_trajectory[head + j + 1].observation_list)
            seq_next_graph_node_vocabulary.append(_padded_trajectory[head + j + 1].graph_node_vocabulary)
            seq_next_graph_relation_vocabulary.append(_padded_trajectory[head + j + 1].graph_relation_vocabulary)
            seq_next_graph_triplets.append(_padded_trajectory[head + j + 1].graph_triplets)
            seq_next_belief.append(_padded_trajectory[head + j + 1].belief)

            how_long = final - (head + j) + 1 if self.accumulate_reward_from_final else 1
            accumulated_rewards = [self.discount_gamma ** i * _padded_trajectory[head + j + i].reward for i in range(how_long)]
            accumulated_rewards = accumulated_rewards[:1]
            reward = torch.sum(torch.stack(accumulated_rewards))
            seq_reward.append(reward)

        trajectory_mask = trajectory_mask[:sample_history_length]

        return [seq_obs, seq_word_indices, seq_ctrlf_indices, seq_graph_node_vocabulary, seq_graph_relation_vocabulary, seq_graph_triplets, seq_belief,
                seq_reward, seq_next_obs, seq_next_graph_node_vocabulary, seq_next_graph_relation_vocabulary, seq_next_graph_triplets, seq_next_belief, quest, trajectory_mask]

    def _get_batch_of_sequences(self, which_memory, batch_size, sample_history_length, contains_first_step):
        assert sample_history_length > 1
        
        obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward = [], [], [], [], [], [], [], [], []
        next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief = [], [], [], [], []
        trajectory_mask = []
        for _ in range(sample_history_length):
            obs.append([])
            word_indices.append([])
            ctrlf_indices.append([])
            graph_node_vocabulary.append([])
            graph_relation_vocabulary.append([])
            graph_triplets.append([])
            belief.append([])
            reward.append([])
            next_obs.append([])
            next_graph_node_vocabulary.append([])
            next_graph_relation_vocabulary.append([])
            next_graph_triplets.append([])
            next_belief.append([])
            trajectory_mask.append([])

        for _ in range(batch_size):
            t = self._get_single_sequence_transition(which_memory, sample_history_length, contains_first_step)
            if t is None:
                continue
            quest.append(t[13])
            for step in range(sample_history_length):
                obs[step].append(t[0][step])
                word_indices[step].append(t[1][step])
                ctrlf_indices[step].append(t[2][step])
                graph_node_vocabulary[step].append(t[3][step])
                graph_relation_vocabulary[step].append(t[4][step])
                graph_triplets[step].append(t[5][step])
                belief[step].append(t[6][step])
                reward[step].append(t[7][step])
                next_obs[step].append(t[8][step])
                next_graph_node_vocabulary[step].append(t[9][step])
                next_graph_relation_vocabulary[step].append(t[10][step])
                next_graph_triplets[step].append(t[11][step])
                next_belief[step].append(t[12][step])
                trajectory_mask[step].append(t[14][step])

        if len(quest) == 0:
            return None
        return [obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward,
                next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, trajectory_mask]

    def get_batch_of_sequences(self, batch_size, sample_history_length):

        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))

        random_number = self.rng.uniform(low=0.0, high=1.0, size=(1,))
        contains_first_step = random_number[0] < 0.2  # hard coded here. So 5% of the sampled batches will have first step

        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch_of_sequences(self.alpha_memory, from_alpha, sample_history_length, contains_first_step)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch_of_sequences(self.beta_memory, from_beta, sample_history_length, contains_first_step)
        if res_alpha is None and res_beta is None:
            return None, None

        obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward = [], [], [], [], [], [], [], [], []
        next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief = [], [], [], [], []
        trajectory_mask = []
        for _ in range(sample_history_length):
            obs.append([])
            word_indices.append([])
            ctrlf_indices.append([])
            graph_node_vocabulary.append([])
            graph_relation_vocabulary.append([])
            graph_triplets.append([])
            belief.append([])
            reward.append([])
            next_obs.append([])
            next_graph_node_vocabulary.append([])
            next_graph_relation_vocabulary.append([])
            next_graph_triplets.append([])
            next_belief.append([])
            trajectory_mask.append([])

        if res_alpha is not None:
            __obs, __quest, __word_indices, __ctrlf_indices, __graph_node_vocabulary, __graph_relation_vocabulary, __graph_triplets, __belief, __reward, __next_obs, __next_graph_node_vocabulary, __next_graph_relation_vocabulary, __next_graph_triplets, __next_belief, __trajectory_mask = res_alpha
            quest += __quest
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                word_indices[i] += __word_indices[i]
                ctrlf_indices[i] += __ctrlf_indices[i]
                graph_node_vocabulary[i] += __graph_node_vocabulary[i]
                graph_relation_vocabulary[i] += __graph_relation_vocabulary[i]
                graph_triplets[i] += __graph_triplets[i]
                belief[i] += __belief[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_graph_node_vocabulary[i] += __next_graph_node_vocabulary[i]
                next_graph_relation_vocabulary[i] += __next_graph_relation_vocabulary[i]
                next_graph_triplets[i] += __next_graph_triplets[i]
                next_belief[i] += __next_belief[i]
                trajectory_mask[i] += __trajectory_mask[i]

        if res_beta is not None:
            __obs, __quest, __word_indices, __ctrlf_indices, __graph_node_vocabulary, __graph_relation_vocabulary, __graph_triplets, __belief, __reward, __next_obs, __next_graph_node_vocabulary, __next_graph_relation_vocabulary, __next_graph_triplets, __next_belief, __trajectory_mask = res_beta
            quest += __quest
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                word_indices[i] += __word_indices[i]
                ctrlf_indices[i] += __ctrlf_indices[i]
                graph_node_vocabulary[i] += __graph_node_vocabulary[i]
                graph_relation_vocabulary[i] += __graph_relation_vocabulary[i]
                graph_triplets[i] += __graph_triplets[i]
                belief[i] += __belief[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_graph_node_vocabulary[i] += __next_graph_node_vocabulary[i]
                next_graph_relation_vocabulary[i] += __next_graph_relation_vocabulary[i]
                next_graph_triplets[i] += __next_graph_triplets[i]
                next_belief[i] += __next_belief[i]
                trajectory_mask[i] += __trajectory_mask[i]

        for i in range(sample_history_length):
            word_indices[i] = torch.stack(word_indices[i], 0)  # batch
            ctrlf_indices[i] = torch.stack(ctrlf_indices[i], 0)  # batch
            reward[i] = torch.stack(reward[i], 0)  # batch
            trajectory_mask[i] = to_pt(np.array(trajectory_mask[i]), enable_cuda=False, type="float")  # batch
            if isinstance(belief[i][0], torch.Tensor):
                belief[i] = torch.stack(belief[i], 0)  # batch x hid
                next_belief[i] = torch.stack(next_belief[i], 0)  # batch x hid

        return [obs, quest, word_indices, ctrlf_indices, graph_node_vocabulary, graph_relation_vocabulary, graph_triplets, belief, reward,
                next_obs, next_graph_node_vocabulary, next_graph_relation_vocabulary, next_graph_triplets, next_belief, trajectory_mask], contains_first_step

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class QAReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, seed=None):
        # prioritized replay memory
        self.rng = np.random.RandomState(seed)
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_rewards, self.beta_rewards = [], []

    def push(self, is_prior=False, reward=0.0, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            self.alpha_memory.append(qa_transition(*args))
            self.alpha_rewards.append(reward)
            if len(self.alpha_memory) > self.alpha_capacity:
                remove_id = self.rng.randint(self.alpha_capacity)
                self.alpha_memory = self.alpha_memory[:remove_id] + self.alpha_memory[remove_id + 1:]
                self.alpha_rewards = self.alpha_rewards[:remove_id] + self.alpha_rewards[remove_id + 1:]
        else:
            self.beta_memory.append(qa_transition(*args))
            self.beta_rewards.append(reward)
            if len(self.beta_memory) > self.beta_capacity:
                remove_id = self.rng.randint(self.beta_capacity)
                self.beta_memory = self.beta_memory[:remove_id] + self.beta_memory[remove_id + 1:]
                self.beta_rewards = self.beta_rewards[:remove_id] + self.beta_rewards[remove_id + 1:]

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = self.rng.choice(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = self.rng.choice(self.alpha_memory, from_alpha) + self.rng.choice(self.beta_memory, from_beta)
        self.rng.shuffle(res)
        return res

    def avg_rewards(self):
        if len(self.alpha_rewards) == 0 and len(self.beta_rewards) == 0 :
            return 0.0
        return np.mean(self.alpha_rewards + self.beta_rewards)

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)