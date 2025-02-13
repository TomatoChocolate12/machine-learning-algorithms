import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CRNN(nn.Module):
    def __init__(self, kernels, cnn_output_size, image_size, rnn_hidden_size, 
                 num_layers=2, num_classes=53, dropout=0.2, normalize=True):
        super(CRNN, self).__init__()
        self.normalize = normalize
        self.encoder = nn.ModuleList()
        for i in range(1, len(kernels)):
            in_channels = kernels[i - 1][0]
            out_channels = kernels[i][0]
            kernel_size = kernels[i][1]
            stride = kernels[i][2]

            self.encoder.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
            if self.normalize:
                self.encoder.append(nn.BatchNorm2d(out_channels))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(2, 2))

        self.cnn = nn.Sequential(*self.encoder)

        # Compute the encoder output size dynamically
        self.encoder_out = self._compute_encoder_output(image_size)
        self.fc = nn.Linear(self.encoder_out, cnn_output_size)
        self.rnn = nn.RNN(
            input_size=cnn_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu',
            dropout=dropout
        )
        if self.normalize:
            self.batch_norm = nn.BatchNorm1d(rnn_hidden_size)
        self.fc_out = nn.Linear(rnn_hidden_size, num_classes)

    def _compute_encoder_output(self, image_size):
        """Helper function to compute CNN output size."""
        dummy_input = torch.zeros(1, *image_size)  # (batch_size, channels, height, width)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        return output.numel()  # Flattened size of CNN output

    def forward(self, x, max_length=12):
        batch_size = x.size(0)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, -1)  # Flatten
        cnn_features = self.fc(cnn_features)
        cnn_features = cnn_features.unsqueeze(1).repeat(1, max_length, 1)
        rnn_output, _ = self.rnn(cnn_features)
        output = self.fc_out(rnn_output)
        return output

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            batch = 0
            for images, targets, lengths in train_loader:
                batch += 1
                optimizer.zero_grad()
                # print("1")
                # Forward pass
                outputs = self.forward(images, max_length=max(lengths))
                outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs

                # Pack targets to match valid tokens
                packed_targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True, enforce_sorted=True).data

                # Slice outputs to match packed targets
                valid_outputs = outputs[:packed_targets.size(0)]

                # Compute loss
                loss = criterion(valid_outputs, packed_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if batch % 100 == 0:
                    print("loss = ", loss.item(), "in batch: ", batch)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


    def evaluate_model(self, loader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets, lengths in loader:
                outputs = self.forward(images, max_length=max(lengths))
                _, predicted = outputs.max(2)
                total += sum(lengths).item()
                correct += (predicted[:, :max(lengths)] == targets).sum().item()

        print(f"Accuracy: {correct / total:.2%}")