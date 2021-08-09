import json
import os
import codecs


def process_file(file_name):
    data = json.load(codecs.open(file_name, 'r', 'utf-8'))
    data = data["data"]
    print("in total there are ", len(data), " documents")
    train = data[:-23]
    valid = data[-23:]

    return train, valid


train, valid = process_file("squad1.1/train-1.1.json")

output_dir = "squad_split"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with codecs.open(output_dir + "/squad.1.1.split.train.json", 'w', encoding='utf-8') as json_file:
    json.dump({"data": train}, json_file, ensure_ascii=False)
with codecs.open(output_dir + "/squad.split.valid.json", 'w', encoding='utf-8') as json_file:
    json.dump({"data": valid}, json_file, ensure_ascii=False)

os.system("cp squad1.1/dev-1.1.json " + output_dir + "/squad.1.1.split.test.json")
