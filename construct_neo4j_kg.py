import json
import re
import sys
import numpy as np
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
import csv

graph = Graph("bolt://localhost:7688")


kg = []
kgfile = "./data/beijing/kg.txt"
with open(kgfile, 'r', encoding='utf-8') as txt_file:
    for line in txt_file:
        kg.append(line.strip().split('\t'))


# 清空现有图数据
graph.delete_all()

# 创建并导入节点和关系
nodes = {}
for s, p, o in tqdm(kg):
    if s not in nodes:
        nodes[s] = Node("Node", name=s)
        graph.create(nodes[s])
    if o not in nodes:
        nodes[o] = Node("Node", name=o)
        graph.create(nodes[o])
    
    # 创建关系并添加到图中
    graph.create(Relationship(nodes[s], p, nodes[o]))

