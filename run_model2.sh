#!/usr/bin/env bash
# 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

data=(slepemapy)
de=(rde sde lsde)
k=(1 5 10 25 100)

for d in ${data[@]}; do
    echo `CUDA_VISIBLE_DEVICES=0 python main.py --model_name sakt --data_name ${d} --de_type none_0 --describe diff_encoding`
    echo `CUDA_VISIBLE_DEVICES=0 python main.py --model_name sakt --data_name ${d} --de_type rde_1000 --describe diff_encoding`
    for e in ${de[@]}; do
        for kk in ${k[@]}; do
            args=(
                --model_name sakt
                --data_name ${d}
                --de_type ${e}_${kk}
                --describe diff_encoding
            )
            echo `CUDA_VISIBLE_DEVICES=1 python main.py ${args[@]}`
        done
    done
done
