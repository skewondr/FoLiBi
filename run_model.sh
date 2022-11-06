#!/usr/bin/env bash
# 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

data=(algebra05 assistments09)

for d in ${data[@]}; do
    args=(
        --model_name sakt
        --data_name ${d}
    )
    echo `CUDA_VISIBLE_DEVICES=0 python main.py ${args[@]}`
done