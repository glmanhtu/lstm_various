import numpy as np
import random
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    """PyTorch dataset class so we can use their Dataloaders."""

    def __init__(self, x, y=None, x_lengths=None):
        self.data = x
        self.labels = y
        self.x_lengths = x_lengths

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.x_lengths is not None and self.labels is not None:
            return self.data[idx], self.labels[idx], self.x_lengths[idx]
        elif self.labels is not None:
            return self.data[idx], self.labels[idx], 0
        else:
            return self.data[idx], 0, 0


def create_dataset(sequence_length, train_percent=0.8):
    """
    Prepare sinosoidal dataset for input into model.
    Data should be in this order: [batch, sequence, feature] where feature is size INPUT_SIZE
    Target labels will the be next value after a sequence of values.
    """

    # Create sin wave at discrete time steps.
    num_time_steps = 1500
    time_steps = np.linspace(start=0, stop=1000, num=num_time_steps, dtype=np.float32)
    discrete_sin_wave = (np.sin(time_steps * 0.5)).reshape(-1, 1)

    # Take (sequence_length + 1) elements & put as a row in sequence_data, extra element is value we want to predict.
    # Move one time step and keep grabbing till we reach the end of our sampled sin wave.
    sequence_data = []
    x_lengths = []
    for i in range(num_time_steps - sequence_length):
        sequence_data.append(discrete_sin_wave[i: i + sequence_length + 1, 0])
        x_lengths.append(sequence_length)
    sequence_data = np.array(sequence_data)

    # Split for train/val.
    num_total_samples = sequence_data.shape[0]
    num_train_samples = int(train_percent * num_total_samples)

    train_set = sequence_data[:num_train_samples, :]
    train_lengths = x_lengths[:num_train_samples]
    test_set = sequence_data[num_train_samples:, :]
    test_lengths = x_lengths[num_train_samples:]

    print('{} total sequence samples, {} used for training'.format(num_total_samples, num_train_samples))

    # Take off the last element of each row and this will be our target value to predict.
    x_train = train_set[:, :-1][:, :, np.newaxis]
    y_train = train_set[:, -1][:, np.newaxis]
    x_test = test_set[:, :-1][:, :, np.newaxis]
    y_test = test_set[:, -1][:, np.newaxis]

    return x_train, y_train, train_lengths, x_test, y_test, test_lengths


def pad_array(arr, size):
    t = size - len(arr)
    return np.pad(arr, pad_width=(0, t), mode='constant')


def create_dataset_various_length(sequence_length_min, sequence_length_max, train_percent=0.8):
    """
    Prepare sinosoidal dataset for input into model.
    Data should be in this order: [batch, sequence, feature] where feature is size INPUT_SIZE
    Target labels will the be next value after a sequence of values.
    """

    # Create sin wave at discrete time steps.
    num_time_steps = 1500
    time_steps = np.linspace(start=0, stop=1000, num=num_time_steps, dtype=np.float32)
    discrete_sin_wave = (np.sin(time_steps * 0.5)).reshape(-1, 1)

    # Take (sequence_length + 1) elements & put as a row in sequence_data, extra element is value we want to predict.
    # Move one time step and keep grabbing till we reach the end of our sampled sin wave.
    sequence_data = []
    x_lengths = []
    for i in range(len(discrete_sin_wave)):
        seq_len = random.randint(sequence_length_min, sequence_length_max)
        if i + seq_len >= len(discrete_sin_wave):
            break
        sequence = discrete_sin_wave[i: i + seq_len, 0]
        label = discrete_sin_wave[i + seq_len]
        # Padding zeroes
        sequence = pad_array(sequence, sequence_length_max + 1)
        sequence[-1] = label
        sequence_data.append(sequence)
        x_lengths.append(seq_len)

    sequence_data = np.array(sequence_data)

    # Split for train/val.
    num_total_samples = sequence_data.shape[0]
    num_train_samples = int(train_percent * num_total_samples)

    train_set = sequence_data[:num_train_samples, :]
    train_x_lengths = x_lengths[:num_train_samples]
    test_set = sequence_data[num_train_samples:, :]
    test_x_lengths = x_lengths[num_train_samples:]

    print('{} total sequence samples, {} used for training'.format(num_total_samples, num_train_samples))

    # Take off the last element of each row and this will be our target value to predict.
    x_train = train_set[:, :-1][:, :, np.newaxis]
    y_train = train_set[:, -1][:, np.newaxis]
    x_test = test_set[:, :-1][:, :, np.newaxis]
    y_test = test_set[:, -1][:, np.newaxis]

    return x_train, y_train, train_x_lengths, x_test, y_test, test_x_lengths
