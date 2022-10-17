#!/usr/bin/env bash

prob=(0.1 0.3 0.5 0.7 0.9)
neg=(0 1)
data=(algebra05 assistments09)

for d in ${data[@]}; do
    for p in ${prob[@]}; do
        args=(
            --model_name cl4kt
            --data_name ${d}
            --mask_prob 0
            --crop_prob 0
            --permute_prob 0
            --replace_prob ${p}
        )
        echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        args=(
            --model_name cl4kt
            --data_name ${d}
            --mask_prob 0
            --crop_prob 0
            --permute_prob 0
            --replace_prob ${p}
            --choose_cl q_cl
        )
        echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        for n in ${neg[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --mask_prob 0
                --crop_prob 0
                --permute_prob 0
                --replace_prob ${p}
                --negative_prob ${n}
                --choose_cl s_cl
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
    done
done

for d in ${data[@]}; do
    for p in ${prob[@]}; do
        args=(
            --model_name cl4kt
            --data_name ${d}
            --mask_prob 0
            --crop_prob 0
            --permute_prob ${p}
            --replace_prob 0
        )
        echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        args=(
            --model_name cl4kt
            --data_name ${d}
            --mask_prob 0
            --crop_prob 0
            --permute_prob ${p}
            --replace_prob 0
            --choose_cl q_cl
        )
        echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        for n in ${neg[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --mask_prob 0
                --crop_prob 0
                --permute_prob ${p}
                --replace_prob 0
                --negative_prob ${n}
                --choose_cl s_cl
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
    done
done