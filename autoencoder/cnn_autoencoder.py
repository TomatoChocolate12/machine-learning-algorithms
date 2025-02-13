import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

class CnnAutoencoder(nn.Module):
    def __init__(self, learning_rate=0.001, kernel_size=3, n_filters=[32, 64, 20], 
                 optimizer='adam', num_epochs=10, n_dim=20):
        super(CnnAutoencoder, self).__init__()
        
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.optimizer_name = optimizer
        self.num_epochs = num_epochs
        self.n_dim = n_dim
        
        self.padding = kernel_size // 2
        
        # Encoder
        encoder_layers = []
        in_channels = 1
        
        for i, n_filter in enumerate(n_filters):
            encoder_layers.extend([
                nn.Conv2d(in_channels, n_filter, kernel_size=kernel_size, padding=self.padding),
                nn.ReLU()
            ])
            
            if i < len(n_filters) - 1:
                encoder_layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = n_filter
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Compute the size of the flattened feature map
        self.encoded_channels = n_filters[-1]
        self.encoded_height = 28 // (2 ** (len(n_filters) - 1))
        self.encoded_width = 28 // (2 ** (len(n_filters) - 1))
        self.flattened_size = self.encoded_channels * self.encoded_height * self.encoded_width
        
        # Linear layer to map to n_dim dimensions
        self.fc_encode = nn.Linear(self.flattened_size, self.n_dim)
        
        # Linear layer to map from n_dim dimensions back to the flattened size
        self.fc_decode = nn.Linear(self.n_dim, self.flattened_size)
        
        # Decoder
        decoder_layers = []
        reversed_filters = n_filters[::-1]
        
        for i in range(len(reversed_filters) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_filters[i], reversed_filters[i+1],
                                   kernel_size=2, stride=2),
                nn.ReLU()
            ])
        
        decoder_layers.extend([
            nn.Conv2d(reversed_filters[-1], 1, kernel_size=kernel_size, padding=self.padding),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.setup_optimizer()
        
        self.criterion = nn.MSELoss()
    
    def setup_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    
    def encode(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()  # Set the model to evaluation mode

        if isinstance(x, DataLoader):
            encoded_data = []
            labels = []

            with torch.no_grad():  # Disable gradient calculation for inference
                for data, label in x:
                    data = data.to(device)
                    encoded = self.encoder(data)
                    encoded = torch.flatten(encoded, start_dim=1)  # Flatten except the batch dimension
                    encoded = self.fc_encode(encoded)
                    encoded_data.append(encoded.cpu())
                    labels.append(label)

            return torch.cat(encoded_data, dim=0), torch.cat(labels, dim=0)
        else:
            # If input is a Tensor, just run it through the encoder
            x = x.to(device)
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)  # Flatten except the batch dimension
            return self.fc_encode(x)

    def decode(self, x):
        x = self.fc_decode(x)  # Map back to the flattened size
        x = x.view(x.size(0), self.encoded_channels, self.encoded_height, self.encoded_width)  # Correct reshaping
        return self.decoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
    def fit(self, train_loader):
        wandb.init()
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        losses = []
        for epoch in range(self.num_epochs):
            self.train()  # Set model to training mode
            total_loss = 0
            
            for batch_idx, (data, lab) in enumerate(train_loader):
                # Move data to device
                data = data.to(device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self(data)
                
                # Calculate loss
                loss = self.criterion(output, data)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_loader)   
            losses.append(avg_loss)         
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                    f'Train Loss: {avg_loss:.6f}')
            wandb.log({"Training Loss": avg_loss})
        return losses

    def predict(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                output = self(data)
                loss = self.criterion(output, data)
                total_loss += loss.item()
        wandb.log({"Validation loss": total_loss/len(dataloader)})
        return total_loss / len(dataloader)