CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --dataset=beauty_geq8 --segment=8 --hidden_units=400 --type=SASREC --num_epochs=200 --train_dir=default --maxlen=50 --dropout_rate=0.2 --device=cuda