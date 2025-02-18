import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from tqdm import tqdm

import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'
import setproctitle
setproctitle.setproctitle('Emb')

import torch
device = torch.device('cuda')
# device = torch.device('cpu')

dataset = 'beijing'
batch_size = 512
model_path = '/put GTE model path here/'
kg_path = f'./data/{dataset}_data/'

print('Loading model')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.load_state_dict(torch.load(f"GTE_finetune/output/{dataset}/embedding_model_epoch_10_{dataset}.pth"))
print('Finish loading.')



rel2name = {'rel_1': ['POI', 'Has Category Of', 'Category1'],
 'rel_2': ['POI', 'Has Category Of', 'Category2'],
 'rel_3': ['POI', 'Has Category Of', 'Category3'],
 'rel_4': ['Category1', 'Exists In', 'POI'],
 'rel_5': ['Category2', 'Exists In', 'POI'],
 'rel_6': ['Category3', 'Exists In', 'POI'],
 'rel_7': ['Brand', 'Belongs To', 'Category1'],
 'rel_8': ['Brand', 'Belongs To', 'Category2'],
 'rel_9': ['Brand', 'Belongs To', 'Category3'],
 'rel_10': ['Category1', 'Has Brand Of', 'Brand'],
 'rel_11': ['Category2', 'Has Brand Of', 'Brand'],
 'rel_12': ['Category3', 'Has Brand Of', 'Brand'],
 'rel_13': ['Region', 'Has', 'POI'],
 'rel_14': ['POI', 'Locates At', 'Region'],
 'rel_15': ['POI', 'Belongs To', 'BusinessArea'],
 'rel_16': ['BusinessArea', 'Contains', 'POI'],
 'rel_17': ['Category2', 'Is Sub-Category Of', 'Category1'],
 'rel_18': ['Category3', 'Is Sub-Category Of', 'Category1'],
 'rel_19': ['Category3', 'Is Sub-Category Of', 'Category2'],
 'rel_20': ['Category1', 'Is Broad Category Of', 'Category2'],
 'rel_21': ['Category1', 'Is Broad Category Of', 'Category3'],
 'rel_22': ['Category2', 'Is Broad Category Of', 'Category3'],
 'rel_23': ['Region', 'Has Large Population Inflow From', 'Region'],
 'rel_24': ['Region', 'Has Large Population Flow To', 'Region'],
 'rel_25': ['Brand', 'Has Placed Store At', 'Region'],
 'rel_26': ['Region', 'Has Store Of', 'Brand'],
 'rel_27': ['Brand', 'Exists In', 'POI'],
 'rel_28': ['POI', 'Has Brand Of', 'Brand'],
 'rel_29': ['Region', 'Is Served By', 'BusinessArea'],
 'rel_30': ['BusinessArea', 'Serves', 'Region'],
 'rel_31': ['Brand', 'Has Related Brand Of', 'Brand'],
 'rel_32': ['POI', 'Is Competitive With', 'POI'],
 'rel_33': ['Region', 'Is Border By', 'Region'],
 'rel_34': ['Region', 'Is Near By', 'Region'],
 'rel_35': ['Region', 'Has Similar Function With', 'Region']}


# load KG
class Data:
    def __init__(self, data_dir):
        self.reg2id = self.load_reg(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_kg(data_dir)
        self.nreg = len(self.reg2id)

        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('region num={}'.format(len(self.reg2id)))
        print('load finished..')

    def load_reg(self, data_dir):      
        with open(data_dir + 'region2info.json', 'r') as f:
            region2info = json.load(f)

        regions = sorted(region2info.keys(), key = lambda x:x)
        reg2id = dict([(x,i) for i,x in enumerate(regions)])

        return reg2id

    def load_kg(self, data_dir):
        ent2id, rel2id = self.reg2id.copy(), {}
        kg_data_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h,r,t = line.strip().split('\t')
                kg_data_str.append((h,r,t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                ent2id[x]
            except KeyError:
                ent2id[x] = len(ent2id)
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        return ent2id, rel2id, kg_data

d = Data(kg_path)
entities = list(d.ent2id.keys())
relations = list(d.rel2id.keys())
relations = [rel2name[x][1] for x in relations]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



def get_embeddings(text_list):
    batch_dict = tokenizer(text_list, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings

# get entities embedding
entity_embeddings = []
for i in tqdm(range(0, len(entities), batch_size)):
    entity_batch = entities[i:i+batch_size]
    batch_embeddings = get_embeddings(entity_batch)
    batch_embeddings = batch_embeddings.cpu().detach().numpy()
    entity_embeddings.append(batch_embeddings)
entity_embeddings = np.concatenate(entity_embeddings, axis=0)
print('Entity embeddings shape:', entity_embeddings.shape)
relation_embeddings = get_embeddings(relations).cpu().detach().numpy()
print('Relation embeddings shape:', relation_embeddings.shape)

np.save(os.path.join(kg_path, 'entity_embeddings.npy'), entity_embeddings)
np.save(os.path.join(kg_path, 'relation_embeddings.npy'), relation_embeddings)

