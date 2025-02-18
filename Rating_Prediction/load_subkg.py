import json
import re
import sys
import numpy as np
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
import time
import os

# 需要先导入kg，见construct_neo4j_kg.py
def find_paths(metapath):
    # 动态构建 Cypher 查询
    if len(metapath) == 1:
        query = f"""
        MATCH p=(n0)-[:{metapath[0]}]->(n1)
        RETURN n0.name AS n0, n1.name AS n1
        """
    else:
        query = 'MATCH p='
        for i, rel in enumerate(metapath):
            query += f"(n{i})-[:{rel}]->"
        query += f"(n{len(metapath)})"
        query += ' RETURN ' + ', '.join([f'n{i}.name AS n{i}' for i in range(len(metapath) + 1)])
    # 执行查询并打印结果
    graph = Graph("bolt://localhost:7688")
    result = graph.run(query)
    all_paths = []
    for record in result:
        path = tuple(record[f"n{i}"] for i in range(len(metapath) + 1))
        all_paths.append(path)
    return all_paths

def get_subkg(dataset, impact_aspects):
    lookup_path = f'/data1/zhouzhilun/LLM_UrbanKG/neo4j_result/{dataset}/'

    kg = []
    with open('./data/{}_data/kg.txt'.format(dataset), 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            kg.append(line.strip().split('\t'))

    for k, v in impact_aspects.items():
        print("============Aspect:", k)
        sub_kg_name = k
        sub_kg_paths = v
        triplets = set()
        for metapath in sub_kg_paths:
            print('Searching Metapath:', metapath) # ['rel_1', 'rel_4']
            start_time = time.time()

            # 如果查询过，直接从文件中读取
            metapath_str = '-'.join(metapath)
            lookup_file = lookup_path + f'{metapath_str}.txt'
            if os.path.exists(lookup_file):
                with open(lookup_file, 'r', encoding='utf-8') as txt_file:
                    for line in txt_file:
                        triplets.add(tuple(line.strip().split('\t')))
                print("Loaded from lookup file.")
                continue
            
            cur_path_triplets = set()
            # 对包含rel4/5/6的metapath进行拆分
            index = None
            for r in ['rel_4', 'rel_5', 'rel_6', 'rel_10', 'rel_11', 'rel_12', 'rel_27']:
                if r in metapath and metapath[0] != r:
                    index = metapath.index(r)
                    break 
            if index is not None:
                mp1, mp2 = metapath[:index], metapath[index:]
                all_paths1 = find_paths(mp1)
                all_paths2 = find_paths(mp2)
                valid_mid_nodes = set([x[-1] for x in all_paths1]) & set([x[0] for x in all_paths2])
                for path in all_paths1:
                    if path[-1] not in valid_mid_nodes:
                        continue
                    assert len(path) == len(mp1) + 1
                    for i in range(len(mp1)):
                        triplets.add(tuple([path[i], mp1[i], path[i+1]]))
                        cur_path_triplets.add(tuple([path[i], mp1[i], path[i+1]]))
                for path in all_paths2:
                    if path[0] not in valid_mid_nodes:
                        continue
                    assert len(path) == len(mp2) + 1
                    for i in range(len(mp2)):
                        triplets.add(tuple([path[i], mp2[i], path[i+1]]))
                        cur_path_triplets.add(tuple([path[i], mp2[i], path[i+1]]))
            else:
                all_paths = find_paths(metapath)
                print(len(all_paths))
                
                for path in all_paths:  # ['node1', 'node2', 'node3']
                    assert len(path) == len(metapath) + 1
                    for i in range(len(metapath)):
                        triplets.add(tuple([path[i], metapath[i], path[i+1]]))
                        cur_path_triplets.add(tuple([path[i], metapath[i], path[i+1]]))
            print("Time:", time.time() - start_time)

            # 保存查询结果
            with open(lookup_file, 'w', encoding='utf-8') as txt_file:
                for t in cur_path_triplets:
                    txt_file.write('\t'.join(t) + '\n')

        unique_triplets = list(triplets)
        print(len(unique_triplets))
        with open('./data/{}_data/kg_{}.txt'.format(dataset, sub_kg_name), 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(["{}\t{}\t{}".format(t[0], t[1], t[2]) for t in unique_triplets]))


