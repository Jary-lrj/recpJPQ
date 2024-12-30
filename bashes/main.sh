#!/bin/bash

# 定义模型列表
models=("NARM")

# 定义数据集列表及其对应的dropout率
declare -A datasets_dropout=(
    ["beauty_geq4"]=0.5
    ["food_seg4"]=0.2
    ["movies_seg4"]=0.2
)

# 切换到上一级目录
cd ..

# 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 循环遍历每个模型和数据集，并执行命令

for dataset in "${!datasets_dropout[@]}"; do
    for model in "${models[@]}"; do
        dropout=${datasets_dropout[$dataset]}
        
        python main.py \
            --model="$model" \
            --dataset="$dataset" \
            --segment=4 \
            --hidden_units=200 \
            --type=cage \
            --num_epochs=200 \
            --train_dir=default \
            --maxlen=50 \
            --dropout_rate="$dropout" \
            --device=cuda
    done
done