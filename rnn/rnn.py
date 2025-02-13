import torch
import torch.nn as nn

device = "cpu"

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        # Use LSTM instead of RNN for better sequence modeling
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the final time step
        return out

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate):
        criterion = nn.MSELoss()  # Loss for training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Training mode
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(sequences)  # Forward pass
                loss = criterion(outputs.squeeze(), labels)  # Compute loss
                loss.backward()
                optimizer.step()

            # Validation after each epoch
            val_mae = self.evaluate_model(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation MAE: {val_mae:.4f}")

    def evaluate_model(self, loader):
        self.eval()  # Evaluation mode
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for sequences, labels in loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = self(sequences).squeeze()
                total_mae += torch.sum(torch.abs(outputs - labels)).item()
                count += labels.size(0)

        return total_mae / count
