import json
import codecs
import numpy as np
from tqdm import tqdm
validation_size = 10000
test_size = 10000


file_path = "processed_wiki_no_squad.json"
data = json.load(codecs.open(file_path, 'r', 'utf-8'))
assert data["date"] == "20210104"
data = data["data"]

titles = list(data.keys())
print("original data size: " + str(len(titles)))

selected = np.random.choice(titles, validation_size + test_size, replace=False).tolist()
validation_keys = selected[:validation_size]
test_keys = selected[validation_size:]

validation_data = {}
for key in tqdm(validation_keys):
    validation_data[key] = data[key]
    del data[key]

test_data = {}
for key in tqdm(test_keys):
    test_data[key] = data[key]
    del data[key]

print("train data size: " + str(len(data)))
print("validation data size: " + str(len(validation_data)))
print("test data size: " + str(len(test_data)))

with codecs.open("wiki_no_squad_train.json", 'w', encoding='utf-8') as json_file:
    json.dump({"date": "20210104", "data": data}, json_file, ensure_ascii=False)
with codecs.open("wiki_no_squad_validation.json", 'w', encoding='utf-8') as json_file:
    json.dump({"date": "20210104", "data": validation_data}, json_file, ensure_ascii=False)
with codecs.open("wiki_no_squad_test.json", 'w', encoding='utf-8') as json_file:
    json.dump({"date": "20210104", "data": test_data}, json_file, ensure_ascii=False)
