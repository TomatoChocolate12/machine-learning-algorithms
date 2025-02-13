import torch
import torch.nn as nn

class PcaAutoencoder(nn.Module):
    def __init__(self, input_dim, n_components):
        super(PcaAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.n_components = n_components

        # Encoder: a single linear layer to reduce dimensionality
        self.encoder = nn.Linear(input_dim, n_components, bias=False)

        # Decoder: a single linear layer to reconstruct the data
        self.decoder = nn.Linear(n_components, input_dim, bias=False)

        # Loss function: Mean Squared Error
        self.loss_function = nn.MSELoss()

    def fit(self, dataloader):
        # Calculate the eigenvalues and eigenvectors for PCA-like behavior
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # First pass: compute mean
            n_samples = 0
            data_sum = None
            
            for batch, lol in dataloader:
                # print(lol)
                batch = batch.view(batch.size(0), -1).float()
                if data_sum is None:
                    data_sum = torch.zeros(batch.size(1))
                data_sum += batch.sum(dim=0)
                n_samples += batch.size(0)
            
            data_mean = data_sum / n_samples

            # Second pass: compute covariance matrix
            covariance_sum = torch.zeros((self.input_dim, self.input_dim))
            
            for batch, _ in dataloader:
                batch = batch.view(batch.size(0), -1).float()
                X_centered = batch - data_mean
                covariance_sum += torch.mm(X_centered.T, X_centered)

            covariance_matrix = covariance_sum / (n_samples - 1)

            # Perform eigenvalue decomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

            # Sort eigenvectors by descending eigenvalues
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            top_eigenvectors = eigenvectors[:, sorted_indices[:self.n_components]]

            # Initialize encoder and decoder weights
            self.encoder.weight.data = top_eigenvectors.T
            self.decoder.weight.data = top_eigenvectors

    def encode(self, X):
        X = X.view(X.size(0), -1).float()
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        X = X.view(X.size(0), -1).float()
        encoded = self.encode(X)
        reconstructed = self.decoder(encoded)
        loss = self.loss_function(reconstructed, X)
        return reconstructed, loss

    def encode_dataset(self, dataloader):
        """Encode an entire dataset using the DataLoader."""
        self.eval()
        encoded_data = []
        labels = []
        
        with torch.no_grad():
            for batch, label in dataloader:
                encoded = self.encode(batch)
                encoded_data.append(encoded)
                labels.append(label)
        
        # print(encoded_data)
        return torch.cat(encoded_data, dim=0), torch.cat(labels, dim=0)

    def reconstruct_dataset(self, dataloader):
        """Reconstruct an entire dataset and compute the overall loss."""
        self.eval()
        reconstructed_data = []
        total_loss = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch, _ in dataloader:
                reconstructed, loss = self(batch)
                reconstructed_data.append(reconstructed)
                total_loss += loss.item() * batch.size(0)
                n_samples += batch.size(0)
                
        avg_loss = total_loss / n_samples
        return torch.cat(reconstructed_data, dim=0), avg_loss

    def evaluate(self, dataloader):
        """Evaluate the model on a dataset and return metrics."""
        self.eval()
        total_loss = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch, _ in dataloader:
                _, loss = self(batch)
                total_loss += loss.item() * batch.size(0)
                n_samples += batch.size(0)
                
        return total_loss / n_samples