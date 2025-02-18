import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch.nn as nn
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


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

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TransELoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TransELoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        # pos_scores 是正三元组的得分，neg_scores 是负三元组的得分
        loss = torch.sum(torch.clamp(pos_scores - neg_scores + self.margin, min=0))
        return loss

class TransE(torch.nn.Module):
    def __init__(self, d, model_path, **kwargs):
        super(TransE, self).__init__()
        # self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), dim
        # self.E = torch.nn.Embedding(self.ne, self.edim)
        # self.R = torch.nn.Embedding(self.nr, self.edim)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedding_model = AutoModel.from_pretrained(model_path)
        # self.loss = MywPSALoss()
        self.loss = TransELoss()
        self.edim = self.embedding_model.config.hidden_size
        self.bn = torch.nn.BatchNorm1d(self.edim)

        self.ent2id, self.rel2id = d.ent2id, d.rel2id
        self.entities = [k for k, v in self.ent2id.items()]
        self.relations = [k for k, v in self.rel2id.items()]

        self.device = kwargs['device']

    def get_embeddings(self, text_list):
        batch_dict = self.tokenizer(text_list, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.embedding_model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings

    def forward(self, h_idx, r_idx, t_idx):
        h_entities = [self.entities[x] for x in h_idx]
        relations = [self.relations[x] for x in r_idx]
        t_entities = [self.entities[x] for x in t_idx]

        relations = [rel2name[x][1] for x in relations]

        h_emb = self.get_embeddings(h_entities) # [batch, edim]
        r_emb = self.get_embeddings(relations)
        t_emb = self.get_embeddings(t_entities)

        score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1) # [batch]
        return score

        # h_emb, r_emb, t_emb = self.E(h_idx), self.R(r_idx), self.E.weight
        # h_emb, r_emb, t_emb = F.normalize(h_emb, 2, -1), F.normalize(r_emb, 2, -1), F.normalize(t_emb, 2, -1)
        # pred = (h_emb + r_emb).view(h_emb.shape[0], 1, -1) - t_emb.view(1, -1, self.E.weight.shape[1])
        # pred = - torch.norm(pred, p=2, dim=-1)
        # return pred


# class TransE(torch.nn.Module):
#     def __init__(self, d, d1, model_path, **kwargs):
#         super(TransE, self).__init__()
#         # self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), kwargs['edim']
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.embedding_model = AutoModel.from_pretrained(model_path)
#         # d1 = self.embedding_model.config.hidden_size
#         # d2 = d1
#         self.edim = self.embedding_model.config.hidden_size
#         self.loss = MywPSALoss()
#         self.bn = torch.nn.BatchNorm1d(self.edim)

#         self.ent2id, self.rel2id = d.ent2id, d.rel2id
#         self.entities = [k for k, v in self.ent2id.items()]
#         self.relations = [k for k, v in self.rel2id.items()]


#     def forward(self, h_idx, r_idx, t_idx):
#         entities = [self.entities[h_idx], self.entities[t_idx]]
#         relations = [self.relations[r_idx]]

#         batch_dict = self.tokenizer(entities, max_length=512, padding=True, truncation=True, return_tensors='pt')
#         batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
#         outputs = self.embedding_model(**batch_dict)
#         entity_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#         batch_dict = self.tokenizer(relations, max_length=512, padding=True, truncation=True, return_tensors='pt')
#         batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
#         outputs = self.embedding_model(**batch_dict)
#         relation_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#         h_emb, r_emb, t_emb = entity_embeddings[0], relation_embeddings[0], entity_embeddings[1]

#         # h_emb, r_emb, t_emb = self.E(h_idx), self.R(r_idx), self.E(t_idx)
#         h_emb, r_emb, t_emb = F.normalize(h_emb, 2, -1), F.normalize(r_emb, 2, -1), F.normalize(t_emb, 2, -1)
#         pred = (h_emb + r_emb).view(h_emb.shape[0], 1, -1) - t_emb.view(1, -1, self.edim)
#         pred = - torch.norm(pred, p=2, dim=-1)
#         return pred


'''
class MywPSALoss(torch.nn.Module):
    def __init__(self):
        super(MywPSALoss, self).__init__()
        return

    def forward(self, pred, tar, v_weight):
        pred1 = F.softmax(pred, dim=1)
        loss_batch = - v_weight.view(-1, 1) * torch.log(pred1 + 1e-10)
        loss = loss_batch[tar == 1].mean()
        return loss

class InterHT(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(InterHT, self).__init__()
        self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), kwargs['edim']
        self.E = torch.nn.Embedding(self.ne, self.edim)
        self.R = torch.nn.Embedding(self.nr, self.edim)
        self.loss = MywPSALoss()
        self.bn = torch.nn.BatchNorm1d(self.edim)
        self.u = 1
        self.gamma = 6.0
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight)
        torch.nn.init.xavier_normal_(self.R.weight)

    def forward(self, h_idx, r_idx, t_idx):
        head, relation, tail = self.E(h_idx), self.R(r_idx), self.E(t_idx)
        nb, edim = head.shape
        n = tail.shape[0]
        a_head, b_head = torch.chunk(head, 2, dim=1)
        re_head, re_mid = torch.chunk(relation, 2, dim=1)
        a_tail, b_tail = torch.chunk(tail, 2, dim=1)
        # print(re_head.shape)
        e_h = torch.ones_like(b_head)
        e_t = torch.ones_like(b_tail)
        a_head = F.normalize(a_head, 2, -1) # [batch, edim]
        # print(a_head.shape)
        a_tail = F.normalize(a_tail, 2, -1)
        # print(a_tail.shape)
        b_head = F.normalize(b_head, 2, -1)
        b_tail = F.normalize(b_tail, 2, -1)
        b_head = b_head + self.u * e_h  # [nb, edim/2]
        b_tail = b_tail + self.u * e_t  # [n, edim/2]
        a_head0 = a_head.view(nb, 1, edim//2).repeat(1, n, 1) # [nb, n, edim/2]
        b_head0 = b_head.view(nb, 1, edim//2).repeat(1, n, 1)
        c_left = torch.einsum('bik,ik->bik', a_head0, b_tail)
        c_right = torch.einsum('ik,bik->bik', a_tail, b_head0)
        re_mid0 = re_mid.view(nb, 1, re_mid.shape[1])
        score = c_left - c_right + re_mid0       
        # score = a_head * b_tail - a_tail * b_head + re_mid
        score = self.gamma - torch.norm(score, p=1, dim=2)
        return score

class TransE(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(TransE, self).__init__()
        self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), kwargs['edim']
        self.E = torch.nn.Embedding(self.ne, self.edim)
        self.R = torch.nn.Embedding(self.nr, self.edim)
        self.loss = MywPSALoss()
        self.bn = torch.nn.BatchNorm1d(self.edim)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight)
        torch.nn.init.xavier_normal_(self.R.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h_emb, r_emb, t_emb = self.E(h_idx), self.R(r_idx), self.E(t_idx)
        h_emb, r_emb, t_emb = F.normalize(h_emb, 2, -1), F.normalize(r_emb, 2, -1), F.normalize(t_emb, 2, -1)
        # pred = (h_emb + r_emb).view(h_emb.shape[0], 1, -1) + t_emb.view(1, -1, self.E.weight.shape[1])
        pred = (h_emb + r_emb).view(h_emb.shape[0], 1, -1) - t_emb.view(1, -1, self.E.weight.shape[1])
        pred = - torch.norm(pred, p=2, dim=-1)
        return pred

class DistMult(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(DistMult, self).__init__()
        self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), kwargs['edim']
        self.E = torch.nn.Embedding(self.ne, self.edim)
        self.R = torch.nn.Embedding(self.nr, self.edim)
        self.loss = MywPSALoss()
        self.bn = torch.nn.BatchNorm1d(self.edim)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight)
        torch.nn.init.xavier_normal_(self.R.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h_emb, r_emb, t_emb = self.E(h_idx), self.R(r_idx), self.E(t_idx)
        pred = (h_emb * r_emb) @ t_emb.transpose(1, 0)
        return pred

class ComplEx(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(ComplEx, self).__init__()
        self.ne, self.nr, self.edim = len(d.ent2id), len(d.rel2id), kwargs['edim']
        self.E_re = torch.nn.Embedding(self.ne, self.edim)
        self.E_im = torch.nn.Embedding(self.ne, self.edim)
        self.R_re = torch.nn.Embedding(self.nr, self.edim)
        self.R_im = torch.nn.Embedding(self.nr, self.edim)
        self.loss = MywPSALoss()
        self.bn = torch.nn.BatchNorm1d(self.edim)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E_re.weight)
        torch.nn.init.xavier_normal_(self.E_im.weight)
        torch.nn.init.xavier_normal_(self.R_re.weight)
        torch.nn.init.xavier_normal_(self.R_im.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h_re, h_im = self.E_re(h_idx), self.E_im(h_idx)
        r_re, r_im = self.R_re(r_idx), self.R_im(r_idx)
        t_re, t_im = self.E_re(t_idx), self.E_im(t_idx)
        pred = h_re * r_re @ t_re.transpose(1, 0) + h_im * r_re @ t_im.transpose(1, 0) + h_re * r_im @ t_im.transpose(1, 0) - h_im * r_im @ t_re.transpose(1, 0)
        return pred

'''


