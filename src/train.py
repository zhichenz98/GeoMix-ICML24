import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import time
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from model import GCN, GIN_Net, APPNP_Net
from geomix import geomix
from sklearn.model_selection import KFold
from utils import preprocess
import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
## Training parameters
parser.add_argument("--data", type=str, default='MUTAG', choices= 'IMDB-BINARY|IMDB-MULTI|PROTEINS|MUTAG|PTC_MR|MSRC_9')
parser.add_argument("--model", type=str, default='GCN', choices= 'GCN|GIN|APPNP')
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=32)

## Augmentation parameters
parser.add_argument("--augment", type=bool, default=True, help='perform geomix or not')
parser.add_argument("--aug_ratio", type=float, default=0.1, help='number of mixup pairs')
parser.add_argument("--num_graphs", type=int, default=10, help='number of mixup graphs per pair')
parser.add_argument("--num_nodes", type=int, default=20, help='number of nodes in the mixup graph')
parser.add_argument("--alpha_fgw", type=float, default=1.0, help='weight for GW term in FGW distance')
parser.add_argument("--sample_dist", type=str, default='uniform', choices='uniform|beta', help='mixup weight sample distribution')
parser.add_argument("--beta_alpha", type=float, default=5, help='Beta(alpha, beta)')
parser.add_argument("--beta_beta", type=float, default=0.5, help='Beta(alpha, beta)')
parser.add_argument("--uniform_min", type=float, default=0.0, help='Uniform(min,max)')
parser.add_argument("--uniform_max", type=float, default=5e-2, help='Uniform(min,max)')
parser.add_argument("--clip_eps", type=float, default=1e-3, help='threshold to filter out zero columns')

## other arguments
parser.add_argument("--vis_P", type=bool, default=False, help='visualize the permutation matrix')
parser.add_argument("--vis_G", type=bool, default=False, help='visualize the mixup graphs')
parser.add_argument('--cuda', type=int, default=1)
args = parser.parse_args()

args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
random_state = 1234


def mixup_cross_entropy_loss(input, target, size_average=True):
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def train(model, optimizer, train_loader):
    model.train()
    
    train_loss = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(args.device)
        out = model(data)  # Perform a single forward pass.
        loss = mixup_cross_entropy_loss(out, data.y)
        train_loss.append(loss.detach().item())
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    
    return np.mean(train_loss)


def test(model, loader):
    model.eval()

    correct = 0
    n = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        gnd = data.y.argmax(dim=1)
        n = n + len(gnd)
        correct += int((pred == gnd).sum())  # Check against ground-truth labels.
    return correct / n  # Derive ratio of correct predictions.


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


def main(args):
    dataset = TUDataset(root='data/TUDataset', name=args.data)
    dataset = list(dataset)
    dataset, num_classes = preprocess(dataset)

    kf = KFold(n_splits=10, shuffle = True, random_state = random_state)
    acc = []
    
    train_time = []
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        
        train_index, val_index = np.split(train_index, [int(8 / 9 * len(train_index))])
        train_dataset = [dataset[j].to(args.device) for j in train_index]
        test_dataset = [dataset[j].to(args.device) for j in test_index]
        val_dataset = [dataset[j].to(args.device) for j in val_index]
        
        t1 = time.time()
        if args.augment:
            train_dataset = geomix(train_dataset, args)
            print(f'Augmentation time: {time.time()-t1:3f}')
        
        ts = time.time()
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
        
        if args.model == 'GCN':
            model = GCN(dataset[0].num_node_features, args.hidden_dim, num_classes).to(args.device)
        elif args.model == 'GIN':
            model = GIN_Net(dataset[0].num_node_features, args.hidden_dim, num_classes).to(args.device)
        elif args.model == 'APPNP':
            model = APPNP_Net(dataset[0].num_node_features, args.hidden_dim, num_classes).to(args.device)
        else:
            raise KeyError('Invalid model name!')
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        
        early_stopping = EarlyStopping(tolerance=50, min_delta=0.1)
        for epoch in range(1, args.epochs+1):
            train_loss = train(model, optimizer, train_loader)
            train_acc = test(model, train_loader)
            scheduler.step()

            with torch.no_grad(): 
                val_acc = test(model, val_loader)
            early_stopping(train_acc, val_acc)
            if early_stopping.early_stop:
                test_acc = test(model, test_loader)
                print(f'Early breaking!')
                print(f'Fold: {i+1:01d}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                break
            if epoch % 50 == 0:
                test_acc = test(model, test_loader)
                print(f'Fold: {i+1:01d}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        train_time.append(time.time() - ts)
        test_acc = test(model, test_loader)
        acc.append(test_acc)
        
    print('dataset: {}, augmentation: {}, model: {} avg_acc:{:.3f}, std:{:.3f}, time:{:.3f}, std:{:.3f}'.format(args.data, args.augment, args.model, np.mean(acc), np.std(acc), np.mean(train_time), np.std(train_time)))

print(args.data, args.model, args.num_nodes)
main(args)

