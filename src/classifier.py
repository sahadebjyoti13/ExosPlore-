# src/classifier.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Exact Class Labels Mapping
CLASS_NAMES = [
    "Planetary Transit",
    "Eclipsing Binary",
    "Stellar Blend",
    "Stellar Variability",
    "Noise/Unknown",
]


class TransitCNN(nn.Module):
    def __init__(self):
        super(TransitCNN, self).__init__()
        """
        Module 3: 1D-CNN + Attention Classifier
        Exact architecture specified by ISRO Hackathon requirements.
        Input shape: (batch, 1, 200)
        """
        # Conv Block 1: Conv1d(1->16, kernel=7, padding=3) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 2: Conv1d(16->32, kernel=5, padding=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Conv Block 3: Conv1d(32->64, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> AdaptiveAvgPool1d(32)
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)

        # MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        # Sequence layout will be permuted to (batch, 32, 64) -> 32 sequence items of 64 features
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True
        )

        # Fully Connected Block: Flatten -> Linear(2048->128) -> ReLU -> Dropout(0.3) -> Linear(128->5)
        self.fc1 = nn.Linear(64 * 32, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # Apply Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Apply Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Apply Block 3
        x = self.adaptive_pool(F.relu(self.bn3(self.conv3(x))))

        # Current Shape: (batch, 64, 32)
        # Permute to (batch, 32, 64) for attention module matching (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)

        # Multihead Attention forward execution
        attn_output, _ = self.attention(x, x, x)

        # Permute back to structural format (batch, 64, 32)
        x = attn_output.permute(0, 2, 1)

        # Flatten out to shape (batch, 2048)
        x = x.reshape(x.size(0), -1)

        # Fully Connected classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


def predict(model, segment_200pts, device="cpu"):
    """
    Executes an operational model prediction over a 200-point single sequence vector.
    Ensures that segment input scales natively to [0, 1] range inside processing.
    """
    model.eval()

    # Critical Requirement: Min-Max Normalization within the local 200-point segment
    seg_min = np.min(segment_200pts)
    seg_max = np.max(segment_200pts)
    if (seg_max - seg_min) > 0:
        norm_segment = (segment_200pts - seg_min) / (seg_max - seg_min)
    else:
        norm_segment = np.zeros_like(segment_200pts)

    # Convert sequence matrix to appropriate batch tensor shape -> (1, 1, 200)
    tensor_input = (
        torch.tensor(norm_segment, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    class_idx = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[class_idx]
    confidence_score = float(probabilities[class_idx])

    return class_idx, class_name, confidence_score, probabilities


def save_checkpoint(model, path="models/transit_cnn.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model checkpoint successfully stored at: {path}")


def load_checkpoint(model, path="models/transit_cnn.pth", device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at path: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model weights loaded successfully from: {path}")
    return model


if __name__ == "__main__":
    print("Testing Classifier Module Architecture validation via mock execution...")

    # Construct model representation
    model = TransitCNN()

    # Test tensor forward pass verification: batch size = 4, channels = 1, sequence points = 200
    mock_input = torch.randn(4, 1, 200)
    try:
        mock_output = model(mock_input)
        print("\n--- Network Forward Pass Successful! ---")
        print(f"Input Shape: {mock_input.shape}")
        print(f"Output Logits Shape: {mock_output.shape} (Expected: [4, 5])")
    except Exception as e:
        print(f"Error during network forward validation routing:\n{e}")
