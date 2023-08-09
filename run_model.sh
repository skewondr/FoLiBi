# pip install accelerate einops yacs iopath

# execute baselines 
CUDA_VISIBLE_DEVICES=0 python main.py --seed 12405 --model_name sakt --data_name algebra05 --de_type none_0 --gpu_num 0 --server_num 0 --describe baselines

# execute rc 
CUDA_VISIBLE_DEVICES=0 python main.py --seed 12405 --model_name sakt --data_name algebra05 --de_type relative_0 --gpu_num 0 --server_num 0 --describe baselines

# execute mono
CUDA_VISIBLE_DEVICES=0 python main.py --seed 12405 --model_name sakt --data_name algebra05 --de_type monotonic_0 --gpu_num 0 --server_num 0 --describe baselines

# execute alibi
CUDA_VISIBLE_DEVICES=0 python main.py --seed 12405 --model_name sakt --data_name algebra05 --de_type alibi1_0 --gpu_num 0 --server_num 0 --describe baselines

