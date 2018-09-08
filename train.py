import torch
import torch.nn as nn
from data import create_corpus
import numpy as np
from torch.utils.data import DataLoader
from data import PaperDecisionDataset, collate_fn
from model import MultimodalRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus, x_title_train, x_authors_train, y_train, \
x_title_test, x_authors_test, y_test = create_corpus('train_dataset', 'test_dataset')

train_batch_size = 16

train_dataset = PaperDecisionDataset(x_title_train, x_authors_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

test_dataset = PaperDecisionDataset(x_title_test, x_authors_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

rnn_model = MultimodalRNN(vocab_title=len(corpus.dictionary_title.idx2word),
                          vocab_authors=len(corpus.dictionary_authors.idx2word),
                          embed_size=64, hidden_size=256).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

epochs = 50
for epoch in range(epochs):
    rnn_model.train()
    epoch_loss = 0.
    correct_train = 0
    for i, batch in enumerate(train_loader):
        title, authors, decision = batch
        batch_loss = 0.
        model_decision = []

        for idx in range(len(title)):
            title_i, authors_i, decision_i = torch.tensor(title[idx]).to(device),\
                                             torch.tensor(authors[idx]).to(device),\
                                             torch.tensor(decision[idx]).to(device)
            logit = rnn_model(title_i, authors_i, device)
            model_decision.append(torch.argmax(torch.softmax(logit, dim=1)))
            loss = criterion(logit, torch.unsqueeze(decision_i, 0))
            batch_loss += loss
            epoch_loss += loss.item()

        batch_loss = batch_loss / len(title)
        model_decision = torch.tensor(model_decision).numpy().tolist()
        for idx in range(len(model_decision)):
            if model_decision[idx] == decision[idx]:
                correct_train += 1

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    rnn_model.eval()
    with torch.no_grad():
        correct_test = 0
        for i, batch in enumerate(test_loader):
            title, authors, decision = batch
            title, authors, decision = torch.tensor(title[0]).to(device), \
                                       torch.tensor(authors[0]).to(device), \
                                       torch.tensor(decision[0]).to(device)

            logit = rnn_model(title, authors, device)
            loss = criterion(logit, torch.unsqueeze(decision, 0))
            model_decision = torch.argmax(torch.softmax(logit, dim=1))
            if model_decision.item() == decision.item():
                correct_test += 1

        print("epoch {} train loss: {:.4} train acc: {:.4} test acc: {:.4}".format(epoch, epoch_loss / float(len(train_loader) * train_batch_size),
                                                                                   correct_train / float(len(train_loader) * train_batch_size),
                                                                                   correct_test / float(len(test_loader))))
