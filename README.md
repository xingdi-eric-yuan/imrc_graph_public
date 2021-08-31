# Interactive Machine Comprehension with Dynamic Knowledge Graphs
---------------------------------------------------------------------------
Implementation for the EMNLP 2021 paper.

## Dependencies

```sh
apt-get -y update
apt-get install -y unzip zip parallel
conda create -p /tmp/imrc python=3.6 numpy scipy cython nltk
conda activate /tmp/imrc
pip install --upgrade pip
pip install numpy==1.16.2
pip install gym==0.15.4
pip install tqdm pipreqs pyyaml pytz visdom
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install transformers
pip install allennlp
```

## Data Preparation

### Split SQuAD 1.1 and preprocess
The original SQuAD dataset does not provide its test set, we take *23* wiki articles from its training set as our validation set. We then use the SQuAD dev set as our test set.

```sh
# download SQuAD from official website, then
python utils/split_original_squad.py
```

To speed up training, we parse (tokenization and SRL) the dataset in advance. 

```sh
python utils/preproc_squad.py
```
This will result `squad_split/processed_squad.1.1.split.[train/valid/test].json`, which are used in iMRC tasks.


### Preprocess Wikipedia data for self-supervised learning

```sh
python utils/get_wiki_filter_squad.py
python utils/split_wiki_data.py
```

This will result `wiki_without_squad/wiki_without_squad_[train/valid/test].json`, which are used to pre-train the continuous belief graph generator.

## Training

To train the agent equipped with different types of graphs, run:

```sh
# without graph
python main.py configs/imrc_none.yaml

# co-occurrence graph
python main.py configs/imrc_cooccur.yaml

# relative position graph
python main.py configs/imrc_rel_pos.yaml

# SRL graph
python main.py configs/imrc_srl.yaml

# continuous belief graph
# in this setting, we need a pre-trained graph generator.
# we provide our pre-trained graph generator at
# https://drive.google.com/drive/folders/1zZ7C_-xaYsfg2Ms7_BO5n3Qzx69UqMKD?usp=sharing

# one can choose to train their own version by:
python pretrain_observation_infomax.py configs/pretrain_cont_bnelief.yaml
# then using the downloaded/saved model checkpoint
python main.py configs/imrc_cont_belief.yaml
```

To change the task settings/configurations:
```yaml
general:
  naozi_capacity: 1  # capacity of agent's external memory queue (1, 3, 5)
  generate_or_point: "point"  # "qmpoint": q+o_t, "point": q, "generate": vocab
  disable_prev_next: False  # False: Easy Mode, True: Hard Mode

model:
  recurrent: True  # recurrent component described in Section 3.3 and Section 4.Additional Results
```

## Citation

```bibtex
@inproceedings{Yuan2021imrc_graph,
  title={Interactive Machine Comprehension with Dynamic Knowledge Graphs},
  author={Xingdi Yuan},
  year={2021},
  booktitle="EMNLP",
}
```

