import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(ProjectionHead, self).__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        self.projection_head = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection_head(x)

class PretrainModel(nn.Module):
    def __init__(self, base_encoder, projection_head):
        super(PretrainModel, self).__init__()
        self.backbone = base_encoder
        self.projection_head = projection_head

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)
    
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLPHead, self).__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class ClassModel(nn.Module):
    def __init__(self, base_encoder, head):
        super(ClassModel, self).__init__()
        self.backbone = base_encoder
        self.head = head

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.head(x)
        return z
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DINOHead, self).__init__()
        self.ReLU = nn.ReLU()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.ReLU(x))

class DINOModel(nn.Module):
    def __init__(self, base_encoder, proj_head,dino_head):
        super(DINOModel, self).__init__()
        self.backbone = base_encoder
        self.head = proj_head
        self.dino = dino_head

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        y = self.head(x)
        z = self.dino(y)
        return z