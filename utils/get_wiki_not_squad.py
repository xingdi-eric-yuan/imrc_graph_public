import json
import re
import sys
import codecs
import nltk
import csv
from tqdm import tqdm
from transformers import BertTokenizer

csv.field_size_limit(sys.maxsize)

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
squad_titles = set()


def get_squad_titles(file_name):

    data = json.load(codecs.open(file_name, 'r', 'utf-8'))
    data = data["data"]
    for doc in tqdm(data):
        title = doc["title"]
        title = title.lower()
        title = title.replace("_", " ")
        title = title.strip()
        squad_titles.add(title)

# original squad data
get_squad_titles("squad1.1/train-v1.1.json")
get_squad_titles("squad1.1/dev-v1.1.json")

print("there are in total " + str(len(squad_titles)) + " titles")
print(squad_titles)

overlap_titles = set()
data = {}
# wikipedia dump is downloaded from here:
# https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia
with open("wikipedia_utf8_filtered_20pageviews.csv", newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in tqdm(csvreader):
        content = row[1]
        if "  " not in content:
            continue

        title, document = content.split("  ", 1)
        title, document = title.strip(), document.strip()
        title = title.lower()
        if title in squad_titles:
            overlap_titles.add(title)
            continue

        if title in data:
            continue

        sent_list = sentence_tokenizer.tokenize(document)
        tokenized_sent_list = []
        for sent in sent_list:
            sent_word_ids = bert_tokenizer.encode(sent.lower(), add_special_tokens=False)
            tokenized_sent_list.append(sent_word_ids)

        data[title] = tokenized_sent_list


with codecs.open("processed_wiki_no_squad.json", 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False)
