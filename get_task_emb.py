import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import setproctitle
setproctitle.setproctitle('Emb')

import torch
# device = torch.device('cuda')
device = torch.device('cpu')

dataset = 'beijing'
model_path = '/put GTE model path here/'

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

print('Loading model')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.load_state_dict(torch.load(f"GTE_finetune/output/{dataset}/embedding_model_epoch_10_{dataset}.pth"))
print('Finish loading.')


input_texts = [
    "The population prediction task aims to predict the population of urban regions based on urban knowledge graph.",
    "The economic prediction task aims to predict the economic indicators of urban regions based on urban knowledge graph.",
    "The comments prediction task aims to predict the number of reviews of restaurants on review platform in urban regions based on urban knowledge graph.",
    "The rating prediction task aims to predict the average rating of restaurants on review platform in urban regions based on urban knowledge graph."
    ]


# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

print(embeddings.shape)
output = embeddings.cpu().detach().numpy()
print(output.shape)
print(output[:3, :3])
np.save(f'task_embeddings_{dataset}.npy', output)

