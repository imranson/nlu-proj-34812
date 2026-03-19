import torch
import torch.nn as nn

class SiameseBiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, dropout=0.5, num_layers=1, lstm_dropout=0.0, freeze_embeddings=False):
        super(SiameseBiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_matrix.size(1), hidden_dim, batch_first=True, bidirectional=True, 
                            num_layers=num_layers, dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.linear1 = nn.Linear(4 * 2 * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim*2, 1)

    def encode(self, indices, lengths):
        embedded = self.embeddings(indices)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        attention_weights = self.attention(output).squeeze(-1)
        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) >= lengths.unsqueeze(1).to(output.device)
        attention_weights = attention_weights.masked_fill(mask, -1e9)
        attention_weights = torch.softmax(attention_weights, dim=1)
        encoded = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        return encoded

    def forward(self, indices_1, length_1, indices_2, length_2):

        encoded_1 = self.encode(indices_1, length_1)
        encoded_2 = self.encode(indices_2, length_2)
        encoded_1 = self.dropout(encoded_1)
        encoded_2 = self.dropout(encoded_2)

        combined = torch.cat((encoded_1, encoded_2, abs(encoded_1 - encoded_2), encoded_1 * encoded_2), dim=1)
        return self.sigmoid(self.linear2(self.relu(self.linear1(combined))))