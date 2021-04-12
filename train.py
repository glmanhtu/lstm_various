from statistics import mean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RNNDataset, create_dataset, create_dataset_various_length
from model import SimpleRNN, LSTM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.01
BATCH_SIZE = 100
NUM_EPOCHS = 100
SEQUENCE_LENGTH = 50
RNN_TYPE = 'LSTM'  # Either 'RNN' or 'LSTM'

USE_VARIOUS_LENGTH_DATASET = True
USE_VARIOUS_LENGTH_MODEL = True
SEQUENCE_LENGTH_MIN = 25
SEQUENCE_LENGTH_MAX = 65


def train_model(model, dataloader, val_loader, loss_function, optimizer, epochs):
    # Train loop.
    for epoch in range(epochs):
        model.train()
        loss_train = []
        for x_batch, y_batch, x_length in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Run our chosen rnn model.
            output = model(x_batch, x_length)

            # Calculate loss.
            loss = loss_function(output, y_batch)

            # Backprop and perform update step.
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        model.eval()
        loss_eval = []
        for x_batch, y_batch, x_length in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                # Run our chosen rnn model.
                output = model(x_batch, x_length)
                loss = loss_function(output, y_batch)
                loss_eval.append(loss.item())

        print(f'Train loss: {mean(loss_train)}, eval loss: {mean(loss_eval)}')


def main():
    if USE_VARIOUS_LENGTH_DATASET:
        train_x, train_y, train_lengths, test_x, test_y, test_lengths = create_dataset_various_length(
            SEQUENCE_LENGTH_MIN, SEQUENCE_LENGTH_MAX, train_percent=0.8)
    else:
        train_x, train_y, train_lengths, test_x, test_y, test_lengths = create_dataset(sequence_length=SEQUENCE_LENGTH,
                                                                                       train_percent=0.8)

    train_dataset = RNNDataset(train_x, train_y, train_lengths)
    train_dataloder = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = RNNDataset(test_x, test_y, test_lengths)
    val_dataloader = DataLoader(val_dataset, batch_size=50)

    # Define the model, optimizer and loss function.
    if USE_VARIOUS_LENGTH_MODEL:
        rnn = LSTM(CNN_embed_dim=1, h_RNN=4, h_FC_dim=4, drop_p=0.05).to(device)
    else:
        rnn = SimpleRNN(RNN_TYPE, input_size=1, hidden_size=4, num_layers=1).to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_model(rnn, dataloader=train_dataloder, val_loader=val_dataloader, loss_function=loss_function,
                optimizer=optimizer, epochs=NUM_EPOCHS)


if __name__ == '__main__':
    main()
