import torch.nn as nn
import torch

class MultimodalRNN(nn.Module):
    def __init__(self, vocab_title, vocab_authors, embed_size, hidden_size):
        super(MultimodalRNN, self).__init__()

        self.vocab_title = vocab_title
        self.vocab_authors = vocab_authors
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # assume a batch size is 1
        self.batch_size = 1

        self.embed_title = nn.Embedding(self.vocab_title, self.embed_size)
        self.embed_authors = nn.Embedding(self.vocab_authors, self.embed_size)

        self.lstm_title = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.lstm_authors = nn.LSTMCell(self.embed_size, self.hidden_size)

        self.mlp = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, 2))

    def forward(self, x_title, x_authors, device):

        # embed x_title, get last hidden states as a summary of the title
        x_title_embed = self.embed_title(x_title)
        # assume batch size = 1
        x_title_embed = x_title_embed.unsqueeze(0)

        h = torch.zeros(self.batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.batch_size, self.hidden_size).to(device)
        for step in range(x_title_embed.shape[1]):
            h, c = self.lstm_title(x_title_embed[:, step, :], (h, c))
        summary_title = h

        # same goes for x_authors
        x_authors_embed = self.embed_authors(x_authors)
        # assume batch size = 1
        x_authors_embed = x_authors_embed.unsqueeze(0)

        h = torch.zeros(self.batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.batch_size, self.hidden_size).to(device)
        for step in range(x_authors_embed.shape[1]):
            h, c = self.lstm_authors(x_authors_embed[:, step, :], (h, c))
        summary_authors = h

        # concat the summaries and apply mlp
        summary = torch.cat([summary_title, summary_authors], dim=1)
        logit = self.mlp(summary)

        return logit