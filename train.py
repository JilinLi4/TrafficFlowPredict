import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
warnings.filterwarnings("ignore")
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Dataloader import *
from model import MVSTT
from utils import metric
import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR  # example
from opt import WarmUpScheduler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS03", help="the datasets name")
parser.add_argument('--exp_name', type=str, default="snp", help="the datasets name")
parser.add_argument('--exp_dir', type=str, default="./result", help="the datasets name")

parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=12, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=12, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=16, help="Number of training batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--in_dim', type=float, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=float, default=64, help="Embed_size")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
args = parser.parse_args()

if __name__ == "__main__":
    data, adj = load_data(args.dataset)
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    data1 = np.mat(data, dtype=np.float32)
    trainX, trainY, valX, valY, testX, testY, mean, std = preprocess_data(data1, time_len, args.train_rate,
                                                                          args.seq_len, args.pre_len)
    train_data = TensorDataset(trainX, trainY)
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_data = TensorDataset(valX, valY)
    val_dataloader = DataLoader(val_data, batch_size=args.batchsize)

    model = MVSTT(adj, args.in_dim, args.embed_size, args.seq_len, args.pre_len, args.heads, 4, args.dropout)
    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=len(train_dataloader),
                                   warmup_steps=len(train_dataloader) * args.epochs * 0.05,
                                   warmup_start_lr=args.lr / 10,
                                   warmup_mode='linear')
    best_mae = 100
    best_rmse = None
    best_mape = None
    best_epoch = 0

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if os.path.exists(exp_dir) == False:
        os.makedirs(exp_dir)
    log_save_dir = f"{exp_dir}/{args.exp_name}_{ args.pre_len * 5}min_{args.dataset}_mean_std.txt"
    print(f"Experment directory: {exp_dir}")
    
    # write exp information
    config_info = ""
    config_info = config_info + f"exp_name: {args.exp_name} \n"
    config_info = config_info + f"batchsize: {args.batchsize} \n"
    config_info = config_info + f"dropout: {args.dropout} \n"
    config_info = config_info + f"in_dim: {args.in_dim} \n"
    config_info = config_info + f"embed_size: {args.embed_size} \n"
    config_info = config_info + f"epochs: {args.epochs} \n"
    config_info = config_info + f"lr: {args.lr} \n"
    config_info = config_info + f"weight_decay: {args.weight_decay} \n"
    print("Config++++++++++++++")
    print(config_info)
    with open(log_save_dir, "a+", encoding='utf-8') as f:
        f.write(config_info)

    for epoch in range(args.epochs):
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            pre = model(x).reshape(-1, adj.shape[0])
            y = y.reshape(-1, adj.shape[0])

            loss = criterion(pre * std + mean, y)
            if batch % 100 == 0:
                print(f"epoch: {epoch} \tbatch: {batch} \tloss: {loss.item()}\t lr: {lr_scheduler.get_lr()[0]}")
                with open(log_save_dir, "a+") as f:
                    f.write(f"epoch: {epoch} \tbatch: {batch}, \tloss: {loss.item()}\t lr: {lr_scheduler.get_lr()[0]}\n")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()

        model.eval()
        P = []
        L = []
        for x, y in val_dataloader:
            x = x.to(device)
            pre = model(x) * std + mean
            P.append(pre.cpu().detach())
            L.append(y)
        pre = torch.cat(P, 0)
        label = torch.cat(L, 0)

        pre = pre.reshape(-1, adj.shape[0])
        label = label.reshape(-1, adj.shape[0])

        mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
        
        DATANAME = args.dataset
        t = datetime.datetime.now()
        dir = str(t)[0:10]
        file = str(t)[10:-7].replace(":", "-")
        # if not os.path.exists(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir))):
        #     os.makedirs(os.path.join("Model/PEMS/{}/{}".format(DATANAME, dir)))
        # torch.save(model.state_dict(), "Model/PEMS/{}/{}/epoch+{}+time{}.pkl".format(DATANAME, dir, epoch, file))
        if mae < best_mae:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_epoch = epoch
            with open("result/{}_{}min_{}_mean_std.txt".format(args.exp_name, args.pre_len * 5, DATANAME), "a+") as f:
                f.write(f"Best MAE: Epoch: {best_epoch}, MAE: {best_mae}, RMSE: {best_rmse}, MAPE: {best_mape}\n")

        print(f"Current epoch: {epoch}\t MAE: {mae}\t RMSE: {rmse}\t MAPE: {mape}\t WAPE: {wape}")
        print(f"Best MAE at epoch: {best_epoch}\t MAE: {best_mae}\t RMSE: {best_rmse}\t MAPE: {best_mape}")

    print("--"*20)
    print(f"{args.exp_name} Best MAE at epoch: {best_epoch}\t MAE: {best_mae}\t RMSE: {best_rmse}\t MAPE: {best_mape}")
    with open(log_save_dir, "a+") as f:
        f.write(f"Best MAE at epoch: {best_epoch}\t MAE: {best_mae}\t RMSE: {best_rmse}\t MAPE: {best_mape}\n")
