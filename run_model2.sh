#!/usr/bin/env bash
# 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

data=(bridge06 slepemapy)

for d in ${data[@]}; do
    args=(
        --model_name sakt
        --data_name ${d}
        --describe default_opt
    )
    echo `CUDA_VISIBLE_DEVICES=1 python main.py ${args[@]}`
done