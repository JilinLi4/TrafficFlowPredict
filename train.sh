exp_dir="/212022081200003/logs/MS_LayerNormal_LSTM"

python train.py --dataset PEMS04 --dropout 0.0 \
--batchsize 16 --lr 0.0002 --weight_decay 7e-5 \
--exp_name PEMS4_lr2e3 --exp_dir $exp_dir;


python train.py --dataset PEMS03 --dropout 0.0 \
--batchsize 16 --lr 0.0002 --weight_decay 7e-5 \
--exp_name PEMS3_lr2e3 --exp_dir $exp_dir;

python train.py --dataset PEMS07 --dropout 0.0 \
--batchsize  4 --lr 0.0004 --weight_decay 7e-5 \
--exp_name PEMS7_wd7e5 --exp_dir $exp_dir;

python train.py --dataset PEMS08 --dropout 0.0 \
--batchsize 16 --lr 0.0004 --weight_decay 7e-5 \
--exp_name PEMS8_wd7e5 --exp_dir $exp_dir;
