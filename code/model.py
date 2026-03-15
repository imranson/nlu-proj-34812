import torch
import torch.nn as nn

class SiameseBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, dropout=0.5, num_layers=1, lstm_dropout=0.0, freeze_embeddings=False):
        super(SiameseBiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_matrix.size(1), hidden_dim, batch_first=True, bidirectional=True, 
                            num_layers=num_layers, dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(4 * 2 * hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def encode(self, indices, lengths):
        embedded = self.embeddings(indices)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_embedded)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        encoded = torch.cat((forward_hidden, backward_hidden), dim=1)
        return encoded

    def forward(self, indices_1, length_1, indices_2, length_2):

        encoded_1 = self.encode(indices_1, length_1)
        encoded_2 = self.encode(indices_2, length_2)
        encoded_1 = self.dropout(encoded_1)
        encoded_2 = self.dropout(encoded_2)

        combined = torch.cat((encoded_1, encoded_2, abs(encoded_1 - encoded_2), encoded_1 * encoded_2), dim=1)
        return self.sigmoid(self.linear(combined))