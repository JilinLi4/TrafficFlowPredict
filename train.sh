"""
1. 修改参数
2. bash train.sh
3. 记录最好的数据，及参数
4. dropout 0 ~ 1.0          单因素: 0       0.2     0.4     0.6
5. lr      0.0001 ~ 0.001   单因素: 0.0002  0.0004 0.0006 0.0008 0.001
"""
# baseline
#  31.338835          19.36582             0.12788188
# python train.py --dataset PEMS04  --dropout 0.0 --batchsize 16 --lr 0.0001 

# dropout 0.2 lr: 0.0001 
python train.py --dataset PEMS04  --dropout 0.0 --batchsize 16 --lr 0.0001  --exp_name snp_baseline
# python train.py --dataset PEMS04  --dropout 0.2 --batchsize 16 --lr 0.0001