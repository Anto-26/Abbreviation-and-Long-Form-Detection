import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    RNN-based token classifier for abbreviation and long-form detection.

    Best-performing model from experiments (Experiment-4, RMSprop):
      - Test accuracy: 87.38%
      - Weighted F1:   0.8569
      - B-AC F1: 0.63  (abbreviation)
      - B-LF F1: 0.21  (long-form begin)
      - I-LF F1: 0.31  (long-form continuation)
    """

    def __init__(self, input_size: int = 300, hidden_dim: int = 128, output_dim: int = 4, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out)
