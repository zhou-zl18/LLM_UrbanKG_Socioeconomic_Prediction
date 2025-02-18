
This is the code for "Harnessing the Synergy between LLM Agents and Knowledge Graphs for Urban Socioeconomic Prediction"
# Step 1: Embedding LM finetuning on UrbanKG
1. First download embedding LM from https://huggingface.co/thenlper/gte-base, and modify the model_path in GTE_finetune/main.py
2. Finetune GTE model:
```
cd GTE_finetune
bash run.sh
```
3. Get embeddings for entities in UrbanKG:
```
cd ..
bash get_kg_er_emb.sh
```

# Step 2: Single task learning
1. Install neo4j, start the neo4j server and run the following command to load UrbanKG to neo4j server:
```
python construct_neo4j_kg.py
```
2. Get meta-paths and their semantic embeddings
```
bash prepare_metapath_round_1.sh
```
3. Train prediction model on single task.
```
cd Population_Prediction
bash run.sh
```
Similar for other tasks: Economic_Prediction, Comments_Prediction, Rating_Prediction

# Step 3: Cross-task communication
1. Get meta-paths and their semantic embeddings
```
bash prepare_metapath_round_2.sh
```
2. Train prediction model for socioeconomic prediction tasks
```
cd Population_Prediction
bash run.sh
```
Similar for other tasks: Economic_Prediction, Comments_Prediction, Rating_Prediction
