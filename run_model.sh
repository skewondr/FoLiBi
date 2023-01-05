#!/usr/bin/env bash
# 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

data=(algebra05)
de=(rde_100 sde_100)
k=(0.1 0.3 0.5 0.7)

for d in ${data[@]}; do
    for kk in ${k[@]}; do
        echo `CUDA_VISIBLE_DEVICES=0 python main.py --model_name sakt --data_name ${d} --de_type none_0 --describe diff_sparsity --sparsity ${kk}`
        for e in ${de[@]}; do
            args=(
                --model_name sakt
                --data_name ${d}
                --de_type ${e}
                --sparsity ${kk}
                --describe diff_sparsity
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
    done
done

for d in ${data[@]}; do
    for kk in ${k[@]}; do
        echo `CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name ${d} --de_type none_0 --describe diff_sparsity --sparsity ${kk}`
        for e in ${de[@]}; do
            args=(
                --model_name akt
                --data_name ${d}
                --de_type ${e}
                --sparsity ${kk}
                --describe diff_sparsity
            )
            echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
        done
    done
done