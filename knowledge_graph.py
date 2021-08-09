import copy
import numpy as np


class DummyGraph(object):

    def __init__(self):
        pass

    def push_batch(self, *argv):
        pass

    def push_one(self, *argv):
        pass

    def push_batch_question(self, *argv):
        pass

    def get_adjacency_matrix(self, *argv):
        return [None for _ in range(self.batch_size)]

    def get_node_vocabulary(self):
        return [None for _ in range(self.batch_size)]

    def get_relation_vocabulary(self):
        return [None for _ in range(self.batch_size)]

    def get_triplets(self):
        return [None for _ in range(self.batch_size)]

    def get_observable_node_mask(self, *argv):
        return None

    def get(self):
        triplets = self.get_triplets()
        node_vocabulary = self.get_node_vocabulary()
        relation_vocabulary = self.get_relation_vocabulary()
        adj = self.get_adjacency_matrix(triplets)
        return triplets, node_vocabulary, relation_vocabulary, adj

    def reset(self, node_capacity, relation_capacity, batch_size):
        self.batch_size = batch_size


class CooccurGraph(object):

    def __init__(self, stopword_ids, tokenizer):
        self.stopword_ids = stopword_ids
        self.tokenizer = tokenizer
        self.word2id = self.tokenizer.get_vocab()

    def update_node_dict(self, b, new_sentence):
        token_ids = [item for item in new_sentence if item not in self.stopword_ids]
        for tid in token_ids:
            if tid in self.node2id[b]:
                continue
            self.id2node[b].append(tid)
        if len(self.id2node[b]) > self.node_capacity:
            self.id2node[b] = self.id2node[b][-self.node_capacity:]

        # update node2id[b]
        self.node2id[b] = {}
        for i in range(len(self.id2node[b])):
            self.node2id[b][self.id2node[b][i]] = i

    def update_relation_dict(self, b, new_sentence):
        if str(new_sentence) not in self.relation2id[b]:
            self.id2relation[b].append(new_sentence)
        if len(self.id2relation[b]) > self.relation_capacity:
            self.id2relation[b] = self.id2relation[b][-self.relation_capacity:]

        # update relation2id[b]
        self.relation2id[b] = {}
        for i in range(len(self.id2relation[b])):
            self.relation2id[b][str(self.id2relation[b][i])] = i

    def push_batch(self, new_sentence_list, prev_action_list, *argv):
        assert len(new_sentence_list) == len(self.triplets) # batch size
        assert len(new_sentence_list) == len(self.id2node) # batch size
        assert len(new_sentence_list) == len(self.id2relation) # batch size
        for b in range(len(new_sentence_list)):
            self.push_one(b, new_sentence_list[b], prev_action_list[b])

    def push_one(self, b, new_sentence, prev_action, *argv):
        assert b < len(self.triplets)
        if prev_action is not None and prev_action in ["stop"]:
            return
        self.update_node_dict(b, new_sentence)
        self.update_relation_dict(b, new_sentence)

        # update triplets
        triplets = []
        for r in range(len(self.id2relation[b])):
            relation = self.id2relation[b][r]
            token_ids = [item for item in relation if item in self.node2id[b]]
            if len(token_ids) <= 1:
                continue
            for i in range(len(token_ids)):
                for j in range(i, len(token_ids)):
                    triplets.append(np.array([self.node2id[b][token_ids[i]], self.node2id[b][token_ids[j]], r]))
        if len(triplets) > 0:
            self.triplets[b] = np.stack(triplets, axis=0)
        else:
            self.triplets[b] = []

    def push_batch_question(self, question_list, *argv):
        assert len(question_list) == len(self.triplets) # batch size
        assert len(question_list) == len(self.id2node) # batch size
        assert len(question_list) == len(self.id2relation) # batch size
        for b in range(len(question_list)):
            self.push_one(b, question_list[b], None)

    def get_adjacency_matrix(self, triplets, *argv):
        batch_size = len(triplets)
        adj = np.zeros((batch_size, self.relation_capacity, self.node_capacity, self.node_capacity), dtype="float32")

        for b in range(batch_size):
            for t in triplets[b]:
                node1, node2, relation = t[0], t[1], t[2]
                adj[b][relation][node1][node2] = 1.0
                adj[b][relation][node2][node1] = 1.0
        return adj

    def get_node_vocabulary(self):
        return copy.deepcopy(self.id2node)

    def get_relation_vocabulary(self):
        return copy.deepcopy(self.id2relation)

    def get_triplets(self):
        return copy.deepcopy(self.triplets)

    def get_observable_node_mask(self, observation_id_matrix, question_id_matrix=None, node_vocabulary=None):
        if node_vocabulary is None:
            node_vocabulary = self.id2node
        assert observation_id_matrix.size(0) == len(node_vocabulary)
        if question_id_matrix is not None:
            assert question_id_matrix.size(0) == len(node_vocabulary)
        observable_node_mask = np.zeros((observation_id_matrix.size(0), self.node_capacity), dtype="float32")
        for b in range(observation_id_matrix.size(0)):
            node2id = {}
            for i, w in enumerate(node_vocabulary[b]):
                node2id[w] = i
            for w_id in observation_id_matrix[b]:
                if w_id in node2id:
                    observable_node_mask[b][node2id[w_id]] = 1.0
            if question_id_matrix is not None:
                for w_id in question_id_matrix[b]:
                    if w_id in node2id:
                        observable_node_mask[b][node2id[w_id]] = 1.0
        return observable_node_mask

    def get(self):
        triplets = self.get_triplets()
        node_vocabulary = self.get_node_vocabulary()
        relation_vocabulary = self.get_relation_vocabulary()
        adj = self.get_adjacency_matrix(triplets)
        return triplets, node_vocabulary, relation_vocabulary, adj

    def reset(self, node_capacity, relation_capacity, batch_size):
        # capacity is the max number of node/relation an agent can reveal
        self.node_capacity = node_capacity
        self.relation_capacity = relation_capacity
        assert relation_capacity > 1
        self.id2node, self.id2relation = [], []
        self.node2id, self.relation2id = [], []
        self.triplets = []
        for _ in range(batch_size):
            self.id2node.append([])
            self.id2relation.append([])
            self.node2id.append({})
            self.relation2id.append({})
            self.triplets.append([])


class RelativePositionGraph(object):

    def __init__(self, stopword_ids, tokenizer):
        self.stopword_ids = stopword_ids
        self.tokenizer = tokenizer
        self.word2id = self.tokenizer.get_vocab()

    def update_node_dict(self, b, new_sentence):
        if str(new_sentence) not in self.sentence2id[b]:
            self.id2sentence[b].append(new_sentence)
            self.sentence2id[b][str(new_sentence)] = len(self.id2sentence[b]) - 1

        token_ids = [item for item in new_sentence if item not in self.stopword_ids]
        for tid in token_ids:
            if tid in self.node2id[b]:
                continue
            self.id2node[b].append(tid)

        if len(self.id2node[b]) > self.node_capacity:
            self.id2node[b] = self.id2node[b][-self.node_capacity:]
        # update node2id[b]
        self.node2id[b] = {}
        for i in range(len(self.id2node[b])):
            self.node2id[b][self.id2node[b][i]] = i

    def push_batch(self, new_sentence_list, prev_action_list, *argv):
        assert len(new_sentence_list) == len(self.triplets) # batch size
        assert len(new_sentence_list) == len(self.id2node) # batch size
        assert len(new_sentence_list) == len(self.id2relation) # batch size
        for b in range(len(new_sentence_list)):
            self.push_one(b, new_sentence_list[b], prev_action_list[b])

    def push_one(self, b, new_sentence, prev_action, *argv):
        assert b < len(self.triplets)
        if prev_action is not None and prev_action in ["stop"]:
            return
        self.update_node_dict(b, new_sentence)

        # update triplets
        triplets = []
        for s_id in range(len(self.id2sentence[b])):
            sent = self.id2sentence[b][s_id]
            token_ids = [item for item in sent if item in self.node2id[b]]
            if len(token_ids) <= 1:
                continue
            for i in range(len(token_ids)):
                if token_ids[i] not in self.node2id[b]:
                    continue
                for j in range(i, len(token_ids)):
                    if token_ids[j] not in self.node2id[b]:
                        continue
                    dist = min(max(j - i, -self.radius), self.radius)
                    triplets.append(np.array([self.node2id[b][token_ids[i]], self.node2id[b][token_ids[j]], self.relation2id[b][str(dist)]]))
        if len(triplets) > 0:
            self.triplets[b] = np.stack(triplets, axis=0)
        else:
            self.triplets[b] = []

    def push_batch_question(self, question_list, *argv):
        assert len(question_list) == len(self.triplets) # batch size
        assert len(question_list) == len(self.id2node) # batch size
        assert len(question_list) == len(self.id2relation) # batch size
        for b in range(len(question_list)):
            self.push_one(b, question_list[b], None)

    def get_adjacency_matrix(self, triplets, *argv):
        batch_size = len(triplets)
        adj = np.zeros((batch_size, self.relation_capacity, self.node_capacity, self.node_capacity), dtype="float32")

        for b in range(batch_size):
            for t in triplets[b]:
                node1, node2, relation = t[0], t[1], t[2]
                adj[b][relation][node1][node2] = 1.0
                adj[b][self.relation_capacity - 1 - relation][node2][node1] = 1.0
        return adj

    def get_node_vocabulary(self):
        return copy.deepcopy(self.id2node)

    def get_relation_vocabulary(self):
        return copy.deepcopy(self.id2relation)

    def get_triplets(self):
        return copy.deepcopy(self.triplets)

    def get_observable_node_mask(self, observation_id_matrix, question_id_matrix=None, node_vocabulary=None):
        if node_vocabulary is None:
            node_vocabulary = self.id2node
        assert observation_id_matrix.size(0) == len(node_vocabulary)
        if question_id_matrix is not None:
            assert question_id_matrix.size(0) == len(node_vocabulary)
        observable_node_mask = np.zeros((observation_id_matrix.size(0), self.node_capacity), dtype="float32")
        for b in range(observation_id_matrix.size(0)):
            node2id = {}
            for i, w in enumerate(node_vocabulary[b]):
                node2id[w] = i
            for w_id in observation_id_matrix[b]:
                if w_id in node2id:
                    observable_node_mask[b][node2id[w_id]] = 1.0
            if question_id_matrix is not None:
                for w_id in question_id_matrix[b]:
                    if w_id in node2id:
                        observable_node_mask[b][node2id[w_id]] = 1.0
        return observable_node_mask

    def get(self):
        triplets = self.get_triplets()
        node_vocabulary = self.get_node_vocabulary()
        relation_vocabulary = self.get_relation_vocabulary()
        adj = self.get_adjacency_matrix(triplets)
        return triplets, node_vocabulary, relation_vocabulary, adj

    def reset(self, node_capacity, relation_capacity, batch_size):
        # capacity is the max number of node/relation an agent can reveal
        self.node_capacity = node_capacity
        self.relation_capacity = relation_capacity
        assert relation_capacity > 1
        assert relation_capacity % 2 == 1  # odd number
        self.id2node, self.id2relation, self.id2sentence = [], [], []
        self.node2id, self.relation2id, self.sentence2id = [], [], []
        self.triplets = []
        minus_id = self.word2id["-"]
        self.radius = int(relation_capacity / 2)
        tmp = [i for i in range(1, self.radius + 1)][::-1] + [0] + [i for i in range(1, self.radius + 1)]
        tmp = [str(item) for item in tmp]
        relations = copy.deepcopy(tmp)
        for i in range(self.radius):
            relations[i] = "-" + relations[i]
        id2relation = [[self.word2id[item]] for item in tmp]
        for i in range(self.radius):
            id2relation[i] = [minus_id] + id2relation[i]
        relation2id = {}
        for i in range(len(id2relation)):
            relation2id[relations[i]] = i

        for _ in range(batch_size):
            self.id2node.append([])
            self.id2relation.append(id2relation)
            self.node2id.append({})
            self.relation2id.append(relation2id)
            self.sentence2id.append({})
            self.id2sentence.append([])
            self.triplets.append([])


class SRLGraph(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def push_batch(self, new_sentence_list, prev_action_list, new_srl_list):
        assert len(new_sentence_list) == len(self.triplets) # batch size
        assert len(new_sentence_list) == len(self.id2node) # batch size
        assert len(new_sentence_list) == len(self.id2relation) # batch size
        for b in range(len(new_sentence_list)):
            self.push_one(b, new_sentence_list[b], prev_action_list[b], new_srl_list[b])

    def push_one(self, b, new_sentence, prev_action, new_srl):
        assert b < len(self.triplets)
        # if game has stopped
        if prev_action is not None and prev_action in ["stop"]:
            return
        # if we have seen this sentence
        if str(new_sentence) in self.has_seen_this_sentence[b]:
            return
        # if empty srl
        if len(new_srl) == 0:
            return
        self.has_seen_this_sentence[b].add(str(new_sentence))
        sentence_root_node_string = "sentence " + str(len(self.has_seen_this_sentence[b]))
        sentence_root_node = self.tokenizer.encode(sentence_root_node_string, add_special_tokens=False)
        self.id2node[b].append(sentence_root_node)
        sentence_root_node_id = len(self.id2node[b]) - 1
        self.node2id[b][str(sentence_root_node)] = sentence_root_node_id
        self.sentence_roots[b].append(sentence_root_node_id)

        triplets = []
        # connect root node with other root nodes
        for other_root_node_id in self.sentence_roots[b]:
            if other_root_node_id == sentence_root_node_id:
                continue
            triplets.append(np.array([sentence_root_node_id, other_root_node_id, self.relation2id[b]["ROOT-ROOT"]]))

        # multiple srls for this sentence
        for i, labels in enumerate(new_srl):
            # find verb
            verb_id = None
            for j, lb in enumerate(labels):
                relation = lb[0]
                if relation == "V":
                    verb_id = j
                    break
            if verb_id is None:
                continue
            # verb node
            verb_node = labels[verb_id][1]
            if str(verb_node) in self.node2id[b]:
                verb_node_id = self.node2id[b][str(verb_node)]
            else:
                self.id2node[b].append(verb_node)
                verb_node_id = len(self.id2node[b]) - 1
                self.node2id[b][str(verb_node)] = verb_node_id
            triplets.append(np.array([verb_node_id, sentence_root_node_id, self.relation2id[b]["V-ROOT"]]))
            # other nodes
            for j, lb in enumerate(labels):
                if j == verb_id:
                    continue
                relation, node = lb
                if relation not in self.relation2id[b]:
                    relation = "OTHER"
                if str(node) in self.node2id[b]:
                    node_id = self.node2id[b][str(node)]
                else:
                    self.id2node[b].append(node)
                    node_id = len(self.id2node[b]) - 1
                    self.node2id[b][str(node)] = node_id
                triplets.append(np.array([node_id, verb_node_id, self.relation2id[b][relation]]))

        if len(triplets) > 0:
            triplets = np.stack(triplets, axis=0)        
            if len(self.triplets[b]) == 0:
                self.triplets[b] = triplets
            else:
                self.triplets[b] = np.concatenate([self.triplets[b], triplets], axis=0)

    def push_batch_question(self, question_list, question_srl_list):
        assert len(question_list) == len(self.triplets) # batch size
        assert len(question_list) == len(self.id2node) # batch size
        assert len(question_list) == len(self.id2relation) # batch size
        for b in range(len(question_list)):
            self.push_one(b, question_list[b], None, question_srl_list[b])

    def get_adjacency_matrix(self, triplets, node_vocabulary=None, relation_vocabulary=None):
        batch_size = len(triplets)
        adj = np.zeros((batch_size, self.relation_capacity, self.node_capacity, self.node_capacity), dtype="float32")
        for b in range(batch_size):
            if node_vocabulary is None:
                node_vocabulary = self.id2node
            if relation_vocabulary is None:
                relation_vocabulary = self.id2relation
                relation2id = self.relation2id[b]
            else:
                relation2id = {}
                for i, w in enumerate(relation_vocabulary[b]):
                    relation2id[w] = i
            for t in triplets[b]:
                node1, node2, relation = t[0], t[1], t[2]
                if node1 >= self.node_capacity or node2 >= self.node_capacity or relation >= self.relation_capacity:
                    continue
                adj[b][relation][node1][node2] = 1.0
                if relation_vocabulary[b][relation] == "ROOT-ROOT":
                    continue
                inv_relation_id = relation2id[relation_vocabulary[b][relation] + "-inv"]
                adj[b][inv_relation_id][node2][node1] = 1.0
            for i in range(min(len(node_vocabulary[b]), self.node_capacity)):
                adj[b][0][i][i] = 1.0
        return adj

    def get_node_vocabulary(self):
        return copy.deepcopy([item[:self.node_capacity] for item in self.id2node])

    def get_relation_vocabulary(self):
        return copy.deepcopy([item[:self.relation_capacity] for item in self.id2relation])

    def get_triplets(self):
        return copy.deepcopy(self.triplets)

    def get(self):
        triplets = self.get_triplets()
        node_vocabulary = self.get_node_vocabulary()
        relation_vocabulary = self.get_relation_vocabulary()
        adj = self.get_adjacency_matrix(triplets)
        return triplets, node_vocabulary, relation_vocabulary, adj

    def reset(self, node_capacity, relation_capacity, batch_size):
        # capacity is the max number of node/relation an agent can reveal
        self.node_capacity = node_capacity
        self.relation_capacity = relation_capacity
        self.has_seen_this_sentence = []
        self.sentence_roots = []
        self.id2node, self.node2id = [], []
        self.id2relation, self.relation2id = [], []
        id2relation = ["SELF", "ROOT-ROOT", "ARG0", "ARG1", "ARG2", "ARG3", "ARGM-TMP", "ARGM-LOC", "ARGM-DIR", "ARGM-MNR", "ARGM-PRP", "ARGM-CAU", "ARGM-REC", "ARGM-ADV", "ARGM-PRD", "V-ROOT", "OTHER", "ARG0-inv", "ARG1-inv", "ARG2-inv", "ARG3-inv", "ARGM-TMP-inv", "ARGM-LOC-inv", "ARGM-DIR-inv", "ARGM-MNR-inv", "ARGM-PRP-inv", "ARGM-CAU-inv", "ARGM-REC-inv", "ARGM-ADV-inv", "ARGM-PRD-inv", "V-ROOT-inv", "OTHER-inv"]
        relation2id = {}
        for i, w in enumerate(id2relation):
            relation2id[w] = i

        self.triplets = []
        for _ in range(batch_size):
            self.has_seen_this_sentence.append(set())
            self.sentence_roots.append([])
            self.id2node.append([])
            self.id2relation.append(id2relation)
            self.node2id.append({})
            self.relation2id.append(relation2id)
            self.triplets.append([])