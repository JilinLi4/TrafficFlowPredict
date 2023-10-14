# python train.py --dataset PEMS03  --dropout 0.0 --batchsize 32 --lr 0.0008 --weight_decay 7e-5  --exp_name PEMS3_baseline_wd7e5  --exp_dir /212022081200003/logs_snp_dataSet; 
# python train.py --dataset PEMS04  --dropout 0.0 --batchsize 32 --lr 0.0008 --weight_decay 7e-5  --exp_name PEMS4_baseline_wd7e5  --exp_dir /212022081200003/logs_snp_dataSet; 

# python train.py --dataset PEMS07  --dropout 0.0 --batchsize 8 --lr 0.0002 --weight_decay 7e-5  --exp_name PEMS7_batch8_wd7e5_lr2e4  --exp_dir /212022081200003/logs_snp_PEM7; 


# python train.py --dataset PEMS08  --dropout 0.0 --batchsize 32 --lr 0.0006 --weight_decay 7e-5  --exp_name PEMS8_batch32_wd7e5_lr6e4  --exp_dir /212022081200003/logs_snp_PEM8;
# python train.py --dataset PEMS08  --dropout 0.0 --batchsize 32 --lr 0.0004 --weight_decay 7e-5  --exp_name PEMS8_batch32_wd7e5_lr4e4  --exp_dir /212022081200003/logs_snp_PEM8;
# python train.py --dataset PEMS08  --dropout 0.0 --batchsize 32 --lr 0.0002 --weight_decay 7e-5  --exp_name PEMS8_batch32_wd7e5_lr2e4  --exp_dir /212022081200003/logs_snp_PEM8;
# python train.py --dataset PEMS08  --dropout 0.0 --batchsize 32 --lr 0.0001 --weight_decay 7e-5  --exp_name PEMS8_batch32_wd7e5_lr1e4  --exp_dir /212022081200003/logs_snp_PEM8;


python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.00001\
  --exp_name PEMS8_batch8_wd1e5_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;


python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.00003\
  --exp_name PEMS8_batch8_wd3e5_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;

python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.00005\
  --exp_name PEMS8_batch8_wd5e5_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;


  python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.00007\
  --exp_name PEMS8_batch8_wd7e5_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;

  python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.00009\
  --exp_name PEMS8_batch8_wd9e5_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;

  python train.py --dataset PEMS08 \
  --dropout 0.0\
  --batchsize 8\
  --lr 0.0002 \
  --weight_decay 0.0001\
  --exp_name PEMS8_batch8_wd1e4_lr2e4\
  --exp_dir /212022081200003/PEMS8/20230918_PEMS8;


 
