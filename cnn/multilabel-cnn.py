import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class MultilabelCNN(nn.Module):
    def __init__(self, task, conv_activ, dense_activ, conv_layers, pool_size, pooling_method, input_size, optimizer_name, num_classes=10):
        super(MultilabelCNN, self).__init__()