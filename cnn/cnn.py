import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class CNN(nn.Module):
    def __init__(self, task, conv_activ, dense_activ, conv_layers, pool_size, pooling_method, input_size, optimizer_name, num_classes=3):
        super(CNN, self).__init__()

        self.task = task
        self.conv_activ = self.get_activation_function(conv_activ)
        self.dense_activ = self.get_activation_function(dense_activ)
        self.pool_method = self.get_pooling_method(pooling_method)
        self.optimizer_name = optimizer_name
        # Store intermediate feature maps
        self.feature_maps = []

        # Build network with separated conv blocks for easier feature map extraction
        self.conv_blocks = nn.ModuleList()
        in_channels = input_size[0]
        current_size = input_size[1]

        # Build convolutional blocks separately
        for (out_channels, kernel_size) in conv_layers:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                self.get_activation_layer(conv_activ),
                self.pool_method(pool_size)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
            current_size = current_size // pool_size

        # Calculate the flattened size
        self.flattened_size = in_channels * current_size * current_size
        if self.task == "regression":
            num_classes = 1
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            self.get_activation_layer(dense_activ),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def get_activation_function(self, func_name):
        if func_name == "sigmoid":
            return torch.sigmoid
        elif func_name == "tanh":
            return torch.tanh
        elif func_name == "softmax":
            return lambda x: torch.softmax(x, dim=1)
        elif func_name == "relu":
            return torch.relu
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def get_activation_layer(self, func_name):
        if func_name == "sigmoid":
            return nn.Sigmoid()
        elif func_name == "tanh":
            return nn.Tanh()
        elif func_name == "relu":
            return nn.ReLU()
        elif func_name == "softmax":
            return nn.Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def get_pooling_method(self, method_name):
        if method_name == "maxpool":
            return nn.MaxPool2d
        elif method_name == "avgpool":
            return nn.AvgPool2d
        else:
            raise ValueError(f"Unsupported pooling method: {method_name}")
        
    def get_optimizer(self, opt_name, parameters, learning_rate):
        if opt_name == "adam":
            return optim.Adam(parameters, lr=learning_rate)
        elif opt_name == "sgd":
            return optim.SGD(parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def forward(self, x, return_features=False):
        self.feature_maps = []

        # Pass through each conv block and store feature maps
        for block in self.conv_blocks:
            x = block(x)
            if return_features:
                self.feature_maps.append(x.detach().cpu())

        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        if return_features:
            return x, self.feature_maps
        return x

    def fit(self, train_loader, epochs=10, learning_rate=0.001):
        wandb.init()
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "regression":
            self.criterion = nn.MSELoss()

        # Initialize the optimizer with model parameters and learning rate
        self.optimizer = self.get_optimizer(self.optimizer_name, self.parameters(), learning_rate)

        losses = []
        for epoch in range(epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            losses.append(train_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            wandb.log({"Training Loss": train_loss})
            wandb.log({"Training Accuracy": train_acc})

        return losses

    def predict(self, val_loader):
        self.eval()
        correct = 0
        total = 0
        device = next(self.parameters()).device

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        wandb.log({"Validation Accuracy": accuracy})
        # print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
