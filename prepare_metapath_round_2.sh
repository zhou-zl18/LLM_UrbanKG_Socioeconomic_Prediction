#!/bin/bash
directories=("./Population_Prediction" "./Economic_Prediction" "./Comments_Prediction" "./Rating_Prediction" )
dataset="shanghai" 


for i in {2..2}; do
    echo "Running iteration $i"
    for dir in "${directories[@]}"; do

        echo "Entering directory: $dir"
        cd "$dir"

        get metapath
        source activate torch-1.9-py3
        python main_prepare_metapath_2.py --round $i --dataset $dataset

        # get embedding
        cp "./output/${dataset}_output/round_${i}/path2info.json" "../path2info.json"
        cd ..
        source activate torch-2.0.1-cu117-py311
        python get_embedding.py
        rm "path2info.json"
        mv "metapath_embeddings.npy" "$dir/output/${dataset}_output/round_${i}/metapath_embeddings.npy"
    done
done
