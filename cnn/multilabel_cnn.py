import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class MultiLabelCNN(nn.Module):
    def __init__(self, conv_activ, dense_activ, conv_layers, pool_size, pooling_method, input_size, optimizer_name, epochs, learning_rate, num_labels=10, threshold=0.5, dropout_rate=0.5):
        super(MultiLabelCNN, self).__init__()

        self.conv_activ = self.get_activation_function(conv_activ)
        self.dense_activ = self.get_activation_function(dense_activ)
        self.pool_method = self.get_pooling_method(pooling_method)
        self.optimizer_name = optimizer_name
        self.feature_maps = []
        self.threshold = threshold
        self.epochs = epochs
        self.lr = learning_rate

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

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            self.get_activation_layer(dense_activ),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )

    def get_activation_function(self, func_name):
        if func_name == "sigmoid":
            return torch.sigmoid
        elif func_name == "tanh":
            return torch.tanh
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

        # Apply Sigmoid activation to ensure output probabilities for each label
        x = torch.sigmoid(x)
        return x

    def fit(self, train_loader):
        # wandb.init()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
        self.optimizer = self.get_optimizer(self.optimizer_name, self.parameters(), self.lr)
        losses = []

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                labels = (labels == 0).float()  # Assuming binary classification for labels

                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                
                predicted = (outputs >= self.threshold).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            # wandb.log({"Training Loss": train_loss})
            # wandb.log({"Training Accuracy": train_acc})
            losses.append(train_loss)

        return losses

    def predict(self, val_loader):
        self.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        correct_predictions = 0
        total_labels = 0
        corr = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels = (labels == 0).float()

                # Get model outputs and apply a threshold to decide label presence
                outputs = self(images)
                predicted = (outputs >= self.threshold).float()  # Convert probabilities to 0 or 1

                # Count correct predictions for exact match
                correct_predictions += (predicted == labels).all(dim=1).sum().item()
                total_labels += labels.size(0)  # Count number of samples

                # Count Hamming accuracy
                corr += (predicted == labels).sum().item()

        accuracy = 100 * correct_predictions / total_labels
        hamming = (100 * corr) / (total_labels * labels.size(1))  # Total number of labels
        print(f"Hamming Accuracy: {hamming:.2f}%")
        # wandb.log({"Hamming Accuracy": hamming})
        return accuracy
