import copy
import numpy as np
import torch

from agent import CustomAgent
from generic import to_pt


class EnsembleAgent(CustomAgent):

    def get_ranks_greedy(self, obs, infos, input_quest, input_quest_mask, quest_id_list, previous_commands, previous_dynamics, previous_belief):
        with torch.no_grad():
            batch_size = len(obs)

            # update inputs for answerer
            if self.not_finished_yet is None:
                self.not_finished_yet = np.ones((len(obs),), dtype="float32")
                self.naozi.push_batch(copy.deepcopy(obs))
                self.kg.push_batch(copy.deepcopy(obs), previous_commands, [item["srl"] for item in infos])
            else:
                for i in range(batch_size):
                    if self.not_finished_yet[i] == 1.0:
                        self.naozi.push_one(i, copy.deepcopy(obs[i]))
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
            
            # info for replay memory
            for i in range(batch_size):
                if self.prev_actions[-1][i] == "stop":
                    self.prev_step_is_still_interacting[i] = 0.0

            # previous step is still interacting, this is because DQN requires one step extra computation
            replay_info = [to_pt(self.prev_step_is_still_interacting, False, "float")]
            return action_rank, ctrlf_rank, ctrlf_word_mask, current_dynamics, current_belief, replay_info

    def get_qa_ranks_greedy(self, observation_list, quest_list, belief):
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
            return point_rank, mask
