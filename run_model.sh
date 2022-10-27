#!/usr/bin/env bash
# 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

data=(assistments09)
cl=(both q_cl)
prob=(0 1.0.3 0.5 0.7 0.9)

for d in ${data[@]}; do
    for c in ${cl[@]}; do
        for p in ${prob[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --choose_cl ${c}
                --mask_prob ${p}
                --crop_prob 0
                --permute_prob 0
                --replace_prob 0
                --negative_prob 1.0
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
        for p in ${prob[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --choose_cl ${c}
                --mask_prob 0
                --crop_prob ${p}
                --permute_prob 0
                --replace_prob 0
                --negative_prob 1.0
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
        for p in ${prob[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --choose_cl ${c}
                --mask_prob 0
                --crop_prob 0
                --permute_prob ${p}
                --replace_prob 0
                --negative_prob 1.0
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
        for p in ${prob[@]}; do
            args=(
                --model_name cl4kt
                --data_name ${d}
                --choose_cl ${c}
                --mask_prob 0
                --crop_prob 0
                --permute_prob 0
                --replace_prob ${p}
                --negative_prob 1.0
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
    done
done