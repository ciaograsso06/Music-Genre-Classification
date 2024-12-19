import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class MusicDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.genres = []
        self._get_data()

    def _get_data(self):
        self.map_genre = {}
        for l, genre in enumerate(os.listdir(self.data_dir)):
            self.map_genre[genre] = l
            genre_dir = os.path.join(self.data_dir, genre)
            self.genres.append(genre)
            for song in os.listdir(genre_dir):
                song_path = os.path.join(genre_dir, song)
                self.data.append(song_path)
                self.labels.append(l)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.genres = np.array(self.genres)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song_path = self.data[idx]
        label = self.labels[idx]
        song = Image.open(song_path)
        im_matrix = np.array(song)
        if im_matrix.shape[2] == 4:
            song = song.convert('RGB')
        im_matrix = np.array(song)
        if self.transform:
            song = self.transform(song)
        # item = {"pixel_values": song, "labels": torch.LongTensor(label)}
        return song, label

def split_dataset(dataset, proportion = [0.7, 0.2, 0.1]):
    train_size = int(proportion[0] * len(dataset))
    val_size = int(proportion[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(dataset, batch_size=32):
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader