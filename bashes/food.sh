cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --model=SASRec --dataset=food_seg4 --segment=4 --hidden_units=200 --type=normal --num_epochs=200 --train_dir=default --maxlen=50 --dropout_rate=0.2  --device=cuda