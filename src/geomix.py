import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import draw_graph, get_class_label
from geomix_utils import lgw, proj_graph


def cand_ind(dataset, num_mixup):
    """select candidate graphs to mixup

    Args:
        dataset (_type_): input graph dataset
        num_mixup (int): number of graphs for mixup

    Returns:
        list: index for mixup graphs
    """
    class_ind = get_class_label(dataset)
    index = []
    g_size = []
    for ele in class_ind:
        tmp_size = np.array([dataset[j].num_nodes for j in ele])
        g_size.append(tmp_size)

    for _ in range(num_mixup):  # select 2 graphs from 2 classes with similar #nodes
        tmp = np.random.choice(np.arange(len(class_ind)), size = 2, replace = False)
        ind1 = np.random.choice(np.arange(len(class_ind[tmp[0]])), size = 1).item()
        ind2 = np.argmin(np.abs(g_size[tmp[0]][ind1] - g_size[tmp[1]]))
        index.append([class_ind[tmp[0]][ind1], class_ind[tmp[1]][ind2]])

    return index


def geomix(dataset, args):
    """Geomix algorithm

    Args:
        dataset (_type_): input graph dataset
        args (_type_): training arguments

    Raises:
        Exception: _description_

    Returns:
        _type_: output graph dataset
    """
    ## randomly select samples and mixup by geomix
    data_out = dataset
    print('Mixup via Low-rank GW...')
    num_mixup = max(int(args.aug_ratio * len(dataset)), 1)
    index = cand_ind(dataset, num_mixup)
    mixup_size = []
    
    for ele in tqdm(index):
        adj1 = to_dense_adj(edge_index = dataset[ele[0]].edge_index, max_num_nodes = dataset[ele[0]].num_nodes).squeeze().to(args.device)
        adj2 = to_dense_adj(edge_index = dataset[ele[1]].edge_index, max_num_nodes = dataset[ele[1]].num_nodes).squeeze().to(args.device)
        x1 = dataset[ele[0]].x.squeeze().to(args.device)
        x2 = dataset[ele[1]].x.squeeze().to(args.device)
        rank = args.num_nodes
        Q, R, g = lgw(adj1, adj2, x1, x2, rank, alpha = args.alpha_fgw)
        
        aug_list = []
        lam_list = []

        # coarsen_adj1, coarsen_x1, _ = proj_graph(Q, adj1, x1)
        # coarsen_adj2, coarsen_x2, _ = proj_graph(R, adj2, x2)
        
        coarsen_adj1, coarsen_adj2, coarsen_x1, coarsen_x2 = proj_graph(Q, R, g, adj1, adj2, x1, x2)
        
        mixup_size.append(coarsen_adj1.shape[0])
        y1 = dataset[ele[0]].y
        y2 = dataset[ele[1]].y
        
        if args.sample_dist == 'uniform':
            lam_list = np.random.uniform(low=args.uniform_min, high=args.uniform_max, size=(args.num_graphs,))
        elif args.sample_dist  == 'beta':
            lam_list = np.random.beta(args.beta_alpha, args.beta_beta, size = (args.num_graphs,))
        else:
            raise Exception('Invalid sampling distribution')
        for i in range(args.num_graphs):
            lam = lam_list[i]
            mixed_adj = (1-lam) * coarsen_adj1 + lam * coarsen_adj2
            mixed_x = (1-lam) * coarsen_x1 + lam * coarsen_x2
            mixed_adj.masked_fill_(mixed_adj.le(args.clip_eps), 0) # mask out edges with small weights
            aug_list.append(mixed_adj)
            edge_index, edge_weight = dense_to_sparse(mixed_adj)
            data_out.append(Data(x = mixed_x, y = (1-lam) * y1 + lam * y2, edge_index = edge_index, edge_weight = edge_weight, num_nodes = mixup_size[-1], edge_attr = None))

        if args.vis_G:  # visulize mixup graphs
            print(lam_list)
            draw_graph([adj1.cpu().numpy(), adj2.cpu().numpy(), coarsen_adj1.cpu().numpy(), coarsen_adj2.cpu().numpy()], title = 'input graphs', thres = 0.0)
            plt.savefig('input.png',format='png',transparent = True)
            draw_graph(aug_list, title = 'mixed graphs', thres = 0.0)
            plt.savefig('mix.png',format='png',transparent = True)
            plt.show()
    print('Average mixup graph size : {:.2f}'.format(np.mean(mixup_size)))
    return data_out
