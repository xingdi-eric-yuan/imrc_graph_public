import json
import re
import codecs
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from transformers import BertTokenizer


sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def get_sent_srl(sent):
    multiple_srls = []
    srl = srl_predictor.predict(sentence=sent)
    for item in srl["verbs"]:
        components = re.findall(r'\[[^]]+\]', item["description"])
        components = [item for item in components if ": " in item]
        components = [item[1:-1] for item in components]
        if len(components) == 0:
            continue
        sent_labels = []
        for c in components:
            tmp1, tmp2 = c.split(": ", 1)
            tmp2_ids = bert_tokenizer.encode(tmp2.lower(), add_special_tokens=False)
            sent_labels.append([tmp1, tmp2_ids])
        multiple_srls.append(sent_labels)
    return multiple_srls


def process_file(file_name):
    processed_data = []

    data = json.load(codecs.open(file_name, 'r', 'utf-8'))
    data = data["data"]
    for doc in tqdm(data):
        for p in doc["paragraphs"]:
            paragraph = p["context"]
            sent_list = sentence_tokenizer.tokenize(paragraph)
            srl_list, tokenized_sent_list = [], []
            for sent in sent_list:
                sent_word_ids = bert_tokenizer.encode(sent.lower(), add_special_tokens=False)
                tokenized_sent_list.append(sent_word_ids)
                multiple_srls = get_sent_srl(sent)
                srl_list.append(multiple_srls)

            for qa in p["qas"]:
                question = qa["question"]
                question_ids = bert_tokenizer.encode(question.lower(), add_special_tokens=False)
                question_srls = get_sent_srl(question)
                answers = [bert_tokenizer.encode(item["text"].lower(), add_special_tokens=False) for item in qa["answers"]]
                dmp = {"sents": tokenized_sent_list, "srls": srl_list, "q": question_ids, "q_srls": question_srls, "a": answers}
                processed_data.append(dmp)
    return processed_data


for split in ["train", "valid", "test"]:
    processed_data = process_file("squad_split/squad.1.1.split." + split + ".json")
    with codecs.open("squad_split/processed_squad.1.1.split." + split + ".json", 'w', encoding='utf-8') as json_file:
        json.dump({"data": processed_data}, json_file, ensure_ascii=False)
