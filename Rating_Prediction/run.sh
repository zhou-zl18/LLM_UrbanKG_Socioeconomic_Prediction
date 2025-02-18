# round 1 best param
# CUDA_VISIBLE_DEVICES=7 python main.py --dataset "beijing" --lr 0.0001 --batch_size 128 --round 1 --save_emb 1
# CUDA_VISIBLE_DEVICES=7 python main.py --dataset "shanghai" --lr 0.0001 --batch_size 64  --round 1 --save_emb 1

# round 2 best param
CUDA_VISIBLE_DEVICES=7 python main.py --dataset "beijing" --lr 0.005 --batch_size 128 --sub_layer 1 --round 2 --save_emb 0
CUDA_VISIBLE_DEVICES=7 python main.py --dataset "shanghai" --lr 0.00005 --batch_size 64 --sub_layer 1 --round 2 --save_emb 0

