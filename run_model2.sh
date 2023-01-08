server=25
gpu=1

for data_name in slepemapy
do
    for model_name in cl4kt
    do
        for de_type in None LSDE_ LRDE_ LRDE RDE
        do
            if [ $de_type = None ]
            then
                CUDA_VISIBLE_DEVICES=${gpu} python main.py --model_name $model_name --data_name $data_name --de_type ${de_type}_0 --gpu_num ${gpu} --server_num ${server}
            elif [ $de_type = LRDE ]
            then
                CUDA_VISIBLE_DEVICES=${gpu} python main.py --model_name $model_name --data_name $data_name --de_type ${de_type}_1000 --gpu_num ${gpu} --server_num ${server}
            elif [ $de_type = RDE ]
            then
                CUDA_VISIBLE_DEVICES=${gpu} python main.py --model_name $model_name --data_name $data_name --de_type ${de_type}_1000 --gpu_num ${gpu} --server_num ${server}
            else
                for k in 1 2 5 10 25 50 100
                do
                    python main.py --model_name $model_name --data_name $data_name --de_type ${de_type}${k} --gpu_num ${gpu} --server_num ${server}
                done
            fi
        done
    done
done
