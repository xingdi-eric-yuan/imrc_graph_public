import os
import copy
import logging
import numpy as np

import torch
import torch.nn.functional as F
from layers import Embedding, masked_softmax, NoisyLinear 
from layers import EncoderBlock, CQAttention, AnswerPointer, StackedRelationalGraphConvolution, AggregationBlock, ObservationDiscriminator
BERT_EMB_SIZE = 1024

logger = logging.getLogger(__name__)


class LSTM_DQN(torch.nn.Module):
    model_name = 'lstm_dqn'

    def __init__(self, config, bert_model, word_vocab, action_space_size):
        super(LSTM_DQN, self).__init__()
        self.config = config
        self.bert_model = bert_model
        self.word_vocab = word_vocab
        self.word_vocab_size = len(self.word_vocab)
        self.action_space_size = action_space_size  # previous, next, stop, ctrl+f <query>
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        self.enable_graph_input = self.config['general']['enable_graph_input']
        # model config
        model_config = self.config['model']
        self.fine_tune_bert = model_config['fine_tune_bert']
        self.bert_encoder = model_config['bert_encoder']
        # graph node
        self.node_embedding_size = model_config['node_embedding_size']
        self.node_embedding_trainable = model_config['node_embedding_trainable']
        self.node_vocab_size = self.config['general']['node_capacity']
        self.relation_embedding_size = model_config['relation_embedding_size']
        self.relation_embedding_trainable = model_config['relation_embedding_trainable']
        self.relation_vocab_size = self.config['general']['relation_capacity']
        self.embedding_dropout = model_config['embedding_dropout']
        
        self.gcn_hidden_dims = model_config['gcn_hidden_dims']
        self.gcn_highway_connections = model_config['gcn_highway_connections']
        self.gcn_num_bases = model_config['gcn_num_bases']
        self.gcn_dropout = model_config['gcn_dropout']
        
        self.gcn_hidden_dims = model_config['gcn_hidden_dims']
        self.gcn_dropout = model_config['gcn_dropout']

        self.encoder_layers = model_config['encoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.aggregation_layers = model_config['aggregation_layers']
        self.aggregation_conv_num = model_config['aggregation_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.attention_dropout = model_config['attention_dropout']
        self.action_scorer_hidden_dim = model_config['action_scorer_hidden_dim']
        self.action_scorer_softmax = model_config['action_scorer_softmax']
        self.question_answerer_hidden_dim = model_config['question_answerer_hidden_dim']
        self.tie_embeddings = model_config['tie_embeddings']
        self.recurrent = model_config['recurrent']

        self.noisy_net = self.config['epsilon_greedy']['noisy_net']

    def _def_layers(self):

        # node embeddings
        self.node_embedding = Embedding(embedding_size=self.node_embedding_size,
                                        vocab_size=self.node_vocab_size,
                                        trainable=self.node_embedding_trainable,
                                        dropout_rate=self.embedding_dropout)
        # relation embeddings
        if self.enable_graph_input in ["srl", "gata"]:
            self.relation_embedding = Embedding(embedding_size=self.relation_embedding_size,
                                                vocab_size=self.relation_vocab_size,
                                                trainable=self.relation_embedding_trainable,
                                                dropout_rate=self.embedding_dropout)
        self.word_embedding_prj = torch.nn.Linear(BERT_EMB_SIZE, self.block_hidden_dim, bias=False)  # BERT_EMB_SIZE is hidden size of bert

        if self.bert_encoder:
            self.encoding_prj = torch.nn.Linear(BERT_EMB_SIZE, self.block_hidden_dim, bias=False)  # BERT_EMB_SIZE is hidden size of bert
        else:
            self.transformer_encoder = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.context_question_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.context_question_attention_resizer = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)
        self.graph_question_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.graph_question_attention_resizer = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)
        self.model_enc_blks = torch.nn.ModuleList([AggregationBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim,
                                                                    k=5, block_hidden_dim=self.block_hidden_dim,
                                                                    n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])
        self.answer_pointer = AnswerPointer(block_hidden_dim=self.block_hidden_dim)

        # node_input_dim, relation_input_dim
        if self.enable_graph_input in ["gata"]:
            node_input_dim = self.node_embedding_size
        else:
            node_input_dim = self.node_embedding_size + self.block_hidden_dim
        if self.enable_graph_input in ["srl", "gata"]:
            relation_input_dim = self.relation_embedding_size
        else:
            relation_input_dim = self.block_hidden_dim
        real_valued_graph = self.enable_graph_input == "gata"

        self.rgcns = StackedRelationalGraphConvolution(entity_input_dim=node_input_dim, relation_input_dim=relation_input_dim, num_relations=self.relation_vocab_size, hidden_dims=self.gcn_hidden_dims, num_bases=self.gcn_num_bases,
        use_highway_connections=self.gcn_highway_connections, dropout_rate=self.gcn_dropout, real_valued_graph=real_valued_graph)
        self.graph_representation_prj = torch.nn.Linear(self.gcn_hidden_dims[-1], self.block_hidden_dim, bias=False)

        if self.recurrent:
            self.rnncell = torch.nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)
            self.dynamics_aggregation = torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim)
        else:
            self.rnncell, self.dynamics_aggregation = None, None

        encoder_output_dim = self.block_hidden_dim

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear

        action_scorer_output_size = 1
        action_scorer_advantage_output_size = self.action_space_size
        action_scorer_ctrlf_output_size = 1
        action_scorer_ctrlf_advantage_output_size = self.word_vocab_size

        self.action_scorer_shared_linear = linear_function(encoder_output_dim, self.action_scorer_hidden_dim)
        action_scorer_input_size = self.action_scorer_hidden_dim

        self.action_scorer_linear = linear_function(action_scorer_input_size, action_scorer_output_size)
        self.action_scorer_linear_advantage = linear_function(action_scorer_input_size, action_scorer_advantage_output_size)

        bert_embedding_size = BERT_EMB_SIZE
        if self.tie_embeddings:
            self.action_scorer_ctrlf = linear_function(action_scorer_input_size, bert_embedding_size)
            self.embedding_to_words = torch.nn.Linear(bert_embedding_size, self.word_vocab_size, bias=False)
            self.embedding_to_words.weight = self.bert_model.embeddings.word_embeddings.weight
            if not self.fine_tune_bert:
                self.embedding_to_words.weight.requires_grad = False
        else:
            self.action_scorer_ctrlf = linear_function(action_scorer_input_size, action_scorer_ctrlf_output_size)
            self.action_scorer_ctrlf_advantage = linear_function(action_scorer_input_size, action_scorer_ctrlf_advantage_output_size)

        # COC pre-training
        self.belief_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.belief_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.belief_linear_1 = torch.nn.Linear(self.block_hidden_dim, self.block_hidden_dim)
        self.belief_linear_2 = torch.nn.Linear(self.block_hidden_dim, int(self.relation_vocab_size / 2) * self.node_vocab_size * self.node_vocab_size)
        self.belief_attention_to_rnn_input = torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim)
        self.belief_graph_rnncell = torch.nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)
        self.observation_discriminator = ObservationDiscriminator(self.block_hidden_dim)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask, node_encodings, node_mask):
        # doc_encodings: batch x doc_len x hid
        # doc_mask: batch x doc_len
        # q_encodings: batch x q_len x hid
        # q_mask: batch x q_len
        # node_encodings: batch x num_node x hid
        # node_mask: batch x num_node
        X = self.context_question_attention(doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)

        if node_encodings is not None:
            graph_rep = self.graph_question_attention(node_encodings, q_encodings, node_mask, q_mask)
            graph_rep = self.graph_question_attention_resizer(graph_rep)
            graph_rep = F.dropout(graph_rep, p=self.block_dropout, training=self.training)
        else:
            graph_rep = None

        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        if node_mask is not None:
            node_mask = torch.bmm(doc_mask.unsqueeze(-1), node_mask.unsqueeze(1))  # batch x time x q_len
        for i in range(self.aggregation_layers):
             M0 = self.model_enc_blks[i](M0, doc_mask, square_mask, graph_rep, node_mask, i * (self.aggregation_conv_num + (3 if graph_rep is not None else 2)) + 1, self.aggregation_layers)
        return M0

    def get_bert_embeddings(self, _input_words, _input_masks):
        # _input_words: batch x time
        # _input_masks: batch x time
        if _input_words.size(1) > 512:  # exceeds the length limit of pre-trained bert max_position_embeddings
            seg_length = 500
            outputs = []
            num_batch = (_input_words.size(1) + seg_length - 1) // seg_length
            for i in range(num_batch):
                batch_input = _input_words[:, i * seg_length: (i + 1) * seg_length]
                batch_mask = _input_masks[:, i * seg_length: (i + 1) * seg_length]
                out = self.get_bert_embeddings(batch_input, batch_mask)
                outputs.append(out)
            return torch.cat(outputs, 1)

        if not self.fine_tune_bert:
            with torch.no_grad():
                res = self.bert_model.embeddings(_input_words)
                res = res * _input_masks.unsqueeze(-1)
        else:
            res = self.bert_model.embeddings(_input_words)
            res = res * _input_masks.unsqueeze(-1)
        return res

    def average_embeddings(self, _input_words, _input_masks):
        # _input_words: batch x time
        # _input_masks: batch x time
        embeddings = self.get_bert_embeddings(_input_words, _input_masks)  # batch x time x emb
        embeddings = self.word_embedding_prj(embeddings)  # batch x time x enc
        embeddings = embeddings * _input_masks.unsqueeze(-1)
        if embeddings.size(1) == 1:
            return embeddings.squeeze(1)
        # masked mean
        return self.masked_mean(embeddings, _input_masks)

    def representation_generator(self, _input_words, _input_masks):
        if self.bert_encoder:
            return self.representation_generator_bert(_input_words, _input_masks)
        else:
            return self.representation_generator_transformer(_input_words, _input_masks)

    def representation_generator_transformer(self, _input_words, _input_masks):
        # _input_words: batch x time
        # _input_masks: batch x time
        if _input_words.size(1) > 512:  # exceeds the length limit of pre-trained bert max_position_embeddings
            seg_length = 500
            outputs = []
            num_batch = (_input_words.size(1) + seg_length - 1) // seg_length
            for i in range(num_batch):
                batch_input = _input_words[:, i * seg_length: (i + 1) * seg_length]
                batch_mask = _input_masks[:, i * seg_length: (i + 1) * seg_length]
                out = self.representation_generator_transformer(batch_input, batch_mask)
                outputs.append(out)
            return torch.cat(outputs, 1)

        embeddings = self.get_bert_embeddings(_input_words, _input_masks)  # batch x time x bert_emb
        embeddings = self.word_embedding_prj(embeddings)  # batch x time x enc
        embeddings = embeddings * _input_masks.unsqueeze(-1)
        square_mask = torch.bmm(_input_masks.unsqueeze(-1), _input_masks.unsqueeze(1))  # batch x time x time
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.transformer_encoder[i](encoding_sequence, _input_masks, square_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc    
            encoding_sequence = encoding_sequence * _input_masks.unsqueeze(-1)
        return encoding_sequence

    def representation_generator_bert(self, _input_words, _input_masks):
        # _input_words: batch x time
        # _input_masks: batch x time
        if _input_words.size(1) > 512:  # exceeds the length limit of pre-trained bert max_position_embeddings
            seg_length = 500
            outputs = []
            num_batch = (_input_words.size(1) + seg_length - 1) // seg_length
            for i in range(num_batch):
                batch_input = _input_words[:, i * seg_length: (i + 1) * seg_length]
                batch_mask = _input_masks[:, i * seg_length: (i + 1) * seg_length]
                out = self.representation_generator_bert(batch_input, batch_mask)
                outputs.append(out)
            return torch.cat(outputs, 1)

        if not self.fine_tune_bert:
            with torch.no_grad():
                encoding_sequence = self.bert_model(input_ids=_input_words, attention_mask=_input_masks)[0]  # batch x time x enc
                encoding_sequence = encoding_sequence * _input_masks.unsqueeze(-1)
        else:
            encoding_sequence = self.bert_model(input_ids=_input_words, attention_mask=_input_masks)[0]  # batch x time x enc
        encoding_sequence = encoding_sequence * _input_masks.unsqueeze(-1)
        encoding_sequence = self.encoding_prj(encoding_sequence)  # batch x time x enc
        encoding_sequence = encoding_sequence * _input_masks.unsqueeze(-1)
        return encoding_sequence

    def masked_mean(self, _input, _mask):
        # _input: batch x time x hid
        # _mask: batch x time
        _input = _input * _mask.unsqueeze(-1)
        # masked mean
        avg_input = torch.sum(_input, 1)  # batch x enc
        _m = torch.sum(_mask, -1)  # batch
        tmp = torch.eq(_m, 0).float()  # batch
        if avg_input.is_cuda:
            tmp = tmp.cuda()
        _m = _m + tmp
        avg_input = avg_input / _m.unsqueeze(-1)  # batch x enc
        return avg_input

    def action_scorer(self, state_representation_sequence, mask, ctrlf_word_mask, previous_dynamics):
        # state_representation: batch x time x enc_dim
        # mask: batch x time
        # ctrlf_word_mask: batch x vocab
        current_dynamics = self.masked_mean(state_representation_sequence, mask)
        if self.recurrent:
            current_dynamics = self.rnncell(current_dynamics, previous_dynamics) if previous_dynamics is not None else self.rnncell(current_dynamics)

        state_representation = self.action_scorer_shared_linear(current_dynamics)  # action scorer hidden dim
        state_representation = torch.relu(state_representation)
        a_rank = self.action_scorer_linear(state_representation)  #  batch x 1
        a_rank_advantage = self.action_scorer_linear_advantage(state_representation)  # advantage stream  batch x n_vocab
        a_rank = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams
        if self.action_scorer_softmax:
            a_rank = masked_softmax(a_rank, axis=-1)  # batch x n_vocab

        if self.tie_embeddings:
            ctrlf_rank = self.action_scorer_ctrlf(state_representation)  # batch x bert_embedding_size
            ctrlf_rank = torch.tanh(ctrlf_rank)
            ctrlf_rank = self.embedding_to_words(ctrlf_rank)  # batch x n_vocab
        else:
            ctrlf_rank = self.action_scorer_ctrlf(state_representation)  #  batch x 1
            ctrlf_rank_advantage = self.action_scorer_ctrlf_advantage(state_representation)  # advantage stream  batch x n_vocab
            ctrlf_rank_advantage = ctrlf_rank_advantage * ctrlf_word_mask
            ctrlf_rank = ctrlf_rank + ctrlf_rank_advantage - ctrlf_rank_advantage.mean(1, keepdim=True)  # combine streams
            ctrlf_rank = ctrlf_rank * ctrlf_word_mask
        if self.action_scorer_softmax:
            ctrlf_rank = masked_softmax(ctrlf_rank, ctrlf_word_mask, axis=-1)  # batch x n_vocab
        return a_rank, ctrlf_rank, current_dynamics
        
    def answer_question(self, matching_representation_sequence, doc_mask):
        # matching_representation_sequence: batch x doc_len x hid
        # doc_mask: batch x doc_len
        pred = self.answer_pointer(matching_representation_sequence, doc_mask)
        return pred

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_ctrlf.reset_noise()
            self.action_scorer_linear.reset_noise()
            self.action_scorer_linear_advantage.reset_noise()
            if not self.tie_embeddings:
                self.action_scorer_ctrlf_advantage.reset_noise()
