cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --dataset=beauty_geq4 --segment=4 --type=recJPQ --num_epochs=200 --train_dir=default --maxlen=50 --dropout_rate=0.5 --device=cuda