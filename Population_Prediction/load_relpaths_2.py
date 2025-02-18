import json
import os
import random
import re
import time
# import openai
from prompt import *
from utils import *

class Relpaths:
    def __init__(self, data_dir, all_tasks, current_task, dataset, round):
        self.all_tasks = all_tasks
        self.current_task = current_task
        self.dataset = dataset
        self.round = round
        self.task2bestround = self.get_best_round()

        

        triplet2rel = {}
        for k,v in rel2triplet.items():
            triplet = '_'.join(v)
            triplet2rel[triplet] = k
        self.triplet2rel = triplet2rel # {HasCategoryOf: rel_1, ...}
        
        # beijing
        # metapaths = {
        #         "Regional Economic Influence":"Region_Has_POI_HasCategoryOf_Category1_HasBrandOf_Brand_HasPlacedStoreAt_Region",
        #         "Geographical and Functional Similarity":"Region_SimilarFunction_Region_NearBy_Region",
        #         "Brand and Business Area Synergy":"Region_HasStoreOf_Brand_BelongTo_Category1_HasBrandOf_Brand_HasPlacedStoreAt_Region",
        #         "Economic Attraction":"Region_Has_POI_HasCategoryOf_Category1_HasBrandOf_Brand_HasPlacedStoreAt_Region",
        #         "Regional POI and Brand Attraction":"Region_Has_POI_HasBrandOf_Brand_HasPlacedStoreAt_Region",
        #         "Brand Attraction Influence":"Region_Has_POI_HasBrandOf_Brand_HasPlacedStoreAt_Region",
        #          }
        # shanghai
        metapaths = {
                "Category-Driven Brand Influence":"Region_HasStoreOf_Brand_BelongTo_Category1_ExistIn_POI_LocateAt_Region",
                "Business Area Category Impact":"Region_ServedBy_BusinessArea_Contain_POI_HasCategoryOf_Category1_ExistIn_POI_LocateAt_Region",
                "Regional Competition Influence":"Region_Has_POI_Competitive_POI_LocateAt_Region_PopulationFlowTo_Region",
                "Economic Attraction":"Region_Has_POI_HasCategoryOf_Category1_HasBrandOf_Brand_HasPlacedStoreAt_Region",
                "Category-Driven Population Dynamics":"Region_Has_POI_HasCategoryOf_Category1_ExistIn_POI_LocateAt_Region",
                "Brand Attraction Influence":"Region_HasStoreOf_Brand_ExistIn_POI_BelongTo_BusinessArea_Serve_Region",
                 }
        
        metapaths = {
                "Category-Driven Brand Influence":"Region_HasStoreOf_Brand_BelongTo_Category1_ExistIn_POI_LocateAt_Region",
                "Business Area Category Impact":"Region_ServedBy_BusinessArea_Contain_POI_HasCategoryOf_Category1_ExistIn_POI_LocateAt_Region",
                "Regional Competition Influence":"Region_Has_POI_Competitive_POI_LocateAt_Region_PopulationFlowTo_Region",
                "Economic Attraction":"Region_Has_POI_HasCategoryOf_Category1_HasBrandOf_Brand_HasPlacedStoreAt_Region",
                "Category-Driven Population Dynamics":"Region_Has_POI_HasCategoryOf_Category1_ExistIn_POI_LocateAt_Region",
                "Brand Attraction Influence":"Region_HasStoreOf_Brand_ExistIn_POI_BelongTo_BusinessArea_Serve_Region",
                 }
        
        # shanghai round 1
        # metapaths = {
        #         "Regional Population Flow":"Region_PopulationFlowTo_Region_PopulationInflowFrom_Region",
        #         "Brand Influence on Population":"Region_HasStoreOf_Brand_HasPlacedStoreAt_Region",
        #         "Business Area Impact":"Region_ServedBy_BusinessArea_Serve_Region",
        #          }
        

        
        self.impact_aspects = {k:self.mp2rels(v) for k,v in metapaths.items()}
        self.llm_prompt, self.llm_output = '', []
        self.llm_prompt_jl, self.impact_aspects_jl, self.llm_output_jl = '', {}, []
        self.llm_prompt_cl, self.impact_aspects_cl, self.llm_output_cl = '', self.impact_aspects, self.llm_output

        self.path2info = self.get_relpaths_description()

        self.dialogue = {'llm_prompt':self.llm_prompt, 
                            'llm_output':self.llm_output, 
                            'impact_aspects':self.impact_aspects, 
                            'llm_prompt_jl':self.llm_prompt_jl, 
                            'llm_output_jl':self.llm_output_jl, 
                            'impact_aspects_jl':self.impact_aspects_jl, 
                            'llm_prompt_cl':self.llm_prompt_cl, 
                            'llm_output_cl':self.llm_output_cl, 
                            'impact_aspects_cl':self.impact_aspects_cl, 
        }

    def mp2rels(self, metapath):
        relations = []
        metapath = metapath.split('_')
        for i, relname in enumerate(metapath):
            if i%2 != 0:
                triplet = '_'.join([metapath[i-1], metapath[i], metapath[i+1]])
                rel = self.triplet2rel[triplet]
                relations.append(rel)
        return [relations]

    def get_best_round(self):
        task2bestround = {x:0 for x in self.all_tasks}
        if self.round == 1:
            return task2bestround
        for task in self.all_tasks:
            all_round_results = []
            for i in range(1, self.round):
                result_file = f'../{task}/output/{self.dataset}_output/round_{i}/result.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                    metric = result['R2']
                    all_round_results.append((i, metric))
            all_round_results = sorted(all_round_results, key=lambda x: x[1], reverse=True)
            best_round = all_round_results[0][0]
            task2bestround[task] = best_round
        return task2bestround
    
    # def clean_text(self, text):
    #     results = []
    #     pattern1 = r'(\d+\.)\s*([^:]+):'
    #     matches1 = re.findall(pattern1, text)
    #     keys = [match[1].strip().replace(' ', '_') for match in matches1]
    #     pattern2 = r'Relevant relation paths include:\s*(\{[^}]*\})'
    #     matches2 = re.findall(pattern2, text)
    #     string_arrays = [match.strip() for match in matches2]
    #     for array in string_arrays:
    #         # 使用正则表达式提取关系链
    #         pattern = re.compile(r'\((.*?)\)')
    #         matches = pattern.findall(array)
    #         # 将每个匹配项分割为关系列表
    #         result = [match.split(', ') for match in matches]
    #         results.append(result)
    #     impact_aspects = dict(zip(keys, results))

    #     pattern3 = r'\d+\.s*[^:]*:.*Relevant relation paths include:\s*\{[^}]*\}'
    #     llm_output= re.findall(pattern3, text, re.DOTALL)
    #     return impact_aspects, llm_output

    def clean_text(self, text):
        start = text.find('[')
        end = text.rfind(']')
        text_json = text[start:end+1]
        metapaths = json.loads(text_json)
        impact_aspects = {}
        for x in metapaths:
            name, reason, metapath = x['name'], x['reason'], x['metapath'].split('_')
            relations = []
            for i, relname in enumerate(metapath):
                if i%2 != 0:
                    triplet = '_'.join([metapath[i-1], metapath[i], metapath[i+1]])
                    rel = self.triplet2rel[triplet]
                    relations.append(rel)
            impact_aspects[name] = [relations]
        llm_output = [text_json]
        return impact_aspects, llm_output


    def get_relpaths(self, data_dir):

        current_task_best_round = self.task2bestround[self.current_task]
        file = f'./output/{self.dataset}_output/round_{current_task_best_round}/best_LLM_output.json'
        if os.path.exists(file):
            print(f'{self.current_task} round {current_task_best_round} LLM output loaded..')
            with open(file, encoding='utf-8') as f:
                previous_output = json.load(f)       
            reference = "For your reference, here are the most optimal paths we have discovered to date.\n"+"Reference results 1:\n"+previous_output[0]+"\n\n"+"""The content of the output can't be exactly the same as the reference results, it has to be combined with your thinking to output a better answer."""
        else:
            reference = ""
        
        llm_prompt = impact_aspects_prompt.format(reference=reference)
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        # print(impact_aspects) # {'Geographical_Proximity': [['rel_33']],
        # print(llm_output) # ["xxx"]
        
        return llm_prompt, impact_aspects, llm_output
    
    def get_relpaths_jl(self, data_dir):

        other_task_output = ''
        for task in self.all_tasks:
            if task == self.current_task:
                continue
            # task_dir = f'../{task}/output/{self.dataset}_output/'
            # task_LLM_output_file = task_dir + 'best_LLM_output.json'
            task_best_round = self.task2bestround[task]
            task_LLM_output_file = f'../{task}/output/{self.dataset}_output/round_{task_best_round}/best_LLM_output.json'
            if os.path.exists(task_LLM_output_file):
                with open(task_LLM_output_file,encoding='utf-8') as f:
                    previous_output = json.load(f)
                    other_task_output += f'{task} LLM output:\n{previous_output[0]}\n\n'
                print(f'{task} round {task_best_round} LLM output loaded..')
            else:
                print(f'{task} LLM output not found!')
        
        if other_task_output == '':
            reference = ""
        else:
            reference = "As some other tasks about urban regions are highly relevant with the current task, we provide the optimal paths we have discovered for some other tasks for your reference.\n"+ other_task_output +"""The content of the output can't be exactly the same as the reference results, it has to be combined with your thinking to output a better answer. """
                
        
        llm_prompt = impact_aspects_prompt_jl.format(reference=reference)
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        # print(impact_aspects)
        # print(llm_output)
        
        return llm_prompt, impact_aspects, llm_output
    
    def get_relpaths_cl(self):
        str1 = ["{}. {}: {{({})}}".format(i+1,k,", ".join(v[0])) for i, (k, v) in enumerate(self.impact_aspects.items())]
        str2 = ["{}. {}: {{({})}}".format(i+4,k,", ".join(v[0])) for i, (k, v) in enumerate(self.impact_aspects_jl.items())]
        llm_prompt = impact_aspects_prompt_cl.format(reference="\n".join(str1+str2))
        result = run_llm(llm_prompt)
        impact_aspects, llm_output= self.clean_text(result)
        # print(llm_prompt)
        # print(llm_output)
        return llm_prompt, impact_aspects, llm_output
    
    def get_relpaths_description(self):
        path2info = {}
        for k,v in self.impact_aspects_cl.items():
            desc = ''
            for i, rel in enumerate(v[0]):
                if i == 0:
                    desc += f'{rel2name[rel][0]} THAT {rel2name[rel][1]} {rel2name[rel][2]}'
                else:
                    desc += f' THAT {rel2name[rel][1]} {rel2name[rel][2]}'

            path2info[k] = {'Relations':v[0], 'Description':desc}
        
        return path2info