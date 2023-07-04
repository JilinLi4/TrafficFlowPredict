- 启动训练
```shell
bash train.sh
```

- 查看当前显卡状态
```shell
nvidia-smi
```

- 实时查看显卡状态
```shell
# watch 实时查看 
# -n 查看间隔  1 1s 
# xxx 查看命令
watch -n 1 nvidia-smi 
```

- 中断命令

> control + c
