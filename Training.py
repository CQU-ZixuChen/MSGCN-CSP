import torch
import scipy.io as scio
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from CNE_GCN import Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from Dataset import MyGraphDataset
parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.6,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')

args = parser.parse_args(args=[])
args.device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
dataset1 = MyGraphDataset("The name of your graph dataset")
args.num_classes = dataset1.num_classes
args.num_features = dataset1.num_features


def tst(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out_p, out_n, _, _, _, _, _, _, x1, x2, x3, x4, x5 = model(data)
        pred_p = out_p.max(dim=1)[1]
        correct += pred_p.eq(data.y).sum().item()
        loss += F.nll_loss(out_p, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset), out_p, data.y, x1, x2, x3, x4, x5


for n in range(1):
    num_training1 = int(len(dataset1) * 0.5)
    num_val1 = int(len(dataset1) * 0.25)
    num_test1 = len(dataset1) - (num_training1 + num_val1)
    training_set1, validation_set1, test_set1 = random_split(dataset1, [num_training1, num_val1, num_test1])


    train_loader = DataLoader(training_set1, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set1, batch_size=num_test1, shuffle=False)
    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    min_loss = 1e10
    min_acc = 0
    patience = 0
    train_loss1_rec = []
    train_loss2_rec = []
    train_acc_rec = []
    val_loss_rec = []
    val_acc_rec = []
    frozen1 = {"pooln1", "pooln2", "pooln3"}
    frozen2 = {"conv1", "conv2", "conv3", "poolp1", "poolp2", "poolp3", "lin1", "lin2", "lin3", "lin0"}

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            correct = 0.
            data = data.to(args.device)
            out_p, out_n, score_n1, score_p1, score_n2, score_p2, score_n3, score_p3, _, _, _, _, _ = model(data)
            # Training positive pooling layer
            # Activate convolution and positive pooling layers, frozen negative pooling layers
            for name1, param1 in model.named_parameters():
                if name1 in frozen1:
                    param1.requires_grad = False
                if name1 in frozen2:
                    param1.requires_grad = True
            loss1 = F.nll_loss(out_p, data.y)
            pred = out_p.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            train_loss1_rec.append(loss1.item())
            train_acc_rec.append(correct / len(data.y))
            loss1.requires_grad_(True)
            loss1.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Training negative pooling layer
            # Activate negative layers, frozen other layers
            out_p, out_n, score_n1, score_p1, score_n2, score_p2, score_n3, score_p3, _, _, _, _, _ = model(data)
            for name2, param2 in model.named_parameters():
                if name2 in frozen1:
                    param2.requires_grad = True
                if name2 in frozen2:
                    param2.requires_grad = False
            # loss2 is the cosine similarity between positive and negative output
            loss2 = torch.mean(F.cosine_similarity(score_n1, score_p1, dim=0)) \
                      + torch.mean(F.cosine_similarity(score_n2, score_p2, dim=0)) \
                      + torch.mean(F.cosine_similarity(score_n3, score_p3, dim=0))
            train_loss2_rec.append(loss2.item())
            loss2.requires_grad_(True)
            loss2.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_acc, val_loss, out, label, _, _, _, _, _ = tst(model, val_loader)
        print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
        val_loss_rec.append(val_loss)
        val_acc_rec.append(val_acc)
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break

    model = Net(args).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    tst_loader = test_loader
    test_acc, test_loss, out_p, label, x1, x2, x3, x4, x5 = tst(model, tst_loader)
    out_p = out_p.detach()
    x1 = x1.detach()
    x2 = x2.detach()
    x3 = x3.detach()
    x4 = x4.detach()
    x5 = x5.detach()
    print("Test accuarcy:{}".format(test_acc))