for lr in 0.00001 0.00005 0.0001 0.0005 0.001 0.005; do
# for lr in 0.00001 0.00005 0.01; do
    for batch_size in 64 128; do
        for sub_layer in 1 2; do
            CUDA_VISIBLE_DEVICES=6 python main.py --dataset "shanghai" --lr $lr --batch_size $batch_size --sub_layer $sub_layer --round 2
        done
    done
done