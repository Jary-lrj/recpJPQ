cd .. 
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --dataset=movies_geq6 --segment=6 --hidden_units=300 --type=recJPQ --num_epochs=200 --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda