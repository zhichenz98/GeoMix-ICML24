import torch
from torch_geometric.nn import GCNConv, APPNP, global_mean_pool
from torch_geometric.nn.models import GIN
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers = 3):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = Linear(hidden_dim, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, data, emb = False):
        # 1. Obtain node embeddings
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if 'edge_weights' in data:
            edge_weight = data.edge_weights.float()
        else:
            edge_weight = None
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        if emb:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class GIN_Net(torch.nn.Module):
    def __init__(self, num_features=1, num_hidden=32, num_classes=1, num_layers = 3):
        super(GIN_Net, self).__init__()

        self.lin1 = Linear(num_features, num_hidden)
        self.conv1 = GIN(num_hidden, num_hidden, num_layers = num_layers)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.lin2 = Linear(num_hidden, num_classes)

    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if 'edge_weights' in data:
            edge_weight = data.edge_weights.float()
        else:
            edge_weight = None
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.conv1(x, edge_index = edge_index, edge_weight = edge_weight))
        x = self.bn1(x)
        
        x = global_mean_pool(x, batch)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
    

class APPNP_Net(torch.nn.Module):
    def __init__(self, num_features, num_hidden=32, num_classes=1, alpha = 0.2, K = 5, num_layers = 1):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(num_features, num_hidden)
        self.lin2 = Linear(num_hidden, num_classes)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(APPNP(K, alpha))


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if 'edge_weights' in data:
            edge_weight = data.edge_weights.float()
        else:
            edge_weight = None
        
        x = F.relu(self.lin1(x))
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        x = global_mean_pool(x, batch)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)