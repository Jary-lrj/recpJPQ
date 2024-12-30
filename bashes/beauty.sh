#!/bin/bash

# 定义模型列表
models=("GRU4Rec")

# 切换到上一级目录
cd ..

# 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 循环遍历每个模型，并执行命令
for model in "${models[@]}"; do
    python main.py \
        --model=$model \
        --dataset=beauty_geq4 \
        --segment=4 \
        --hidden_units=200 \
        --type=cage \
        --num_epochs=200 \
        --train_dir=default \
        --maxlen=50 \
        --dropout_rate=0.5 \
        --device=cuda
done