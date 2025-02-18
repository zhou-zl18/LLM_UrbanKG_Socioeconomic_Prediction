from load_relpaths_2 import Relpaths
from load_subkg import *
import argparse
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shanghai", help="choose the dataset.")
    parser.add_argument('--current_task', default='Population_Prediction', type=str, help='')
    parser.add_argument('--round', default=2, type=int, help='communication round')
    args = parser.parse_args()
    print(args)

    # all_tasks = ['Population_Prediction']
    # all_tasks = ['Population_Prediction', 'Order_Prediction']
    all_tasks = ['Population_Prediction', 'Economic_Prediction', 'Comments_Prediction', 'Rating_Prediction'] ##########################

    data_dir = f'./data/{args.dataset}_data/'
    output_dir = f'./output/{args.dataset}_output/round_{args.round}/'
    assert os.path.exists(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    r1 = Relpaths(data_dir=data_dir, all_tasks=all_tasks, current_task=args.current_task, dataset=args.dataset, round=args.round)

    get_subkg(args.dataset, r1.impact_aspects_cl)
    
    relpaths = [k for k in r1.impact_aspects_cl.keys()] # ['Geographical_Proximity', 'Business_and_Service...essibility', 'Population_Flow', 'Brand_Presence', 'Functional_Similarity']
    with open(output_dir + 'relpaths.json', 'w') as f:
        json.dump(relpaths, f, indent=4)
    with open(output_dir + 'path2info.json', 'w') as f:
        json.dump(r1.path2info, f, indent=4)

    # with open(output_dir + 'best_LLM_output.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(r1.llm_output+r1.llm_output_jl, json_file, indent=4)
    # with open(output_dir + 'dialogue.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(r1.dialogue, json_file, indent=4)

    