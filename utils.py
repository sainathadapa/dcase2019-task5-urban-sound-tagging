import numpy as np
from RandomErasing import RandomErasing

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision.models
from torchvision import transforms


def prepare_data(df, unknown_to_known):
    df = df.reset_index()
    df['slno'] = df.assign(slno=1).groupby('audio_filename')['slno'].cumsum()
    df.set_index(['audio_filename', 'slno'], inplace=True)

    df_unknown = df.copy().loc[:, list(unknown_to_known.keys())]
    df.drop(columns=list(unknown_to_known.keys()), inplace=True)

    y_mask = df.copy()
    y_mask.loc[:, :] = 1
    for unknown, known in unknown_to_known.items():
        y_mask.loc[
            df_unknown[unknown] > 0.5,
            known
        ] = 0

    df = df.swaplevel(i=1, j=0, axis=0).sort_index()

    y_mask = y_mask.swaplevel(i=1, j=0, axis=0).sort_index()

    y = np.concatenate([
        df.loc[[1], :].values[..., np.newaxis],
        df.loc[[2], :].values[..., np.newaxis],
        df.loc[[3], :].values[..., np.newaxis]
    ], axis=2)

    y_mask = np.concatenate([
        y_mask.loc[[1], :].values[..., np.newaxis],
        y_mask.loc[[2], :].values[..., np.newaxis],
        y_mask.loc[[3], :].values[..., np.newaxis]
    ], axis=2)

    X = np.concatenate([
        np.expand_dims(np.load('./data/logmelspec/{}.npy'.format(x)).T[:635, :], axis=0)
        for x in df.loc[[1], :].reset_index(1).audio_filename.tolist()])
    X = np.expand_dims(X, axis=1)

    return X, y, y_mask


random_erasing = RandomErasing()


class AudioDataset(Dataset):

    def __init__(self, X, y, weights, transform=None):
        self.X = X
        self.y = y
        self.weights = weights
        self.transform = transform
        self.pil = transforms.ToPILImage()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx, ...]

        if self.transform:
            # min-max transformation
            this_min = sample.min()
            this_max = sample.max()
            sample = (sample - this_min) / (this_max - this_min)

            # randomly cycle the file
            i = np.random.randint(sample.shape[1])
            sample = torch.cat([
                sample[:, i:, :],
                sample[:, :i, :]],
                dim=1)

            # apply albumentations transforms
            sample = np.array(self.pil(sample))
            sample = self.transform(image=sample)
            sample = sample['image']
            sample = sample[None, :, :].permute(0, 2, 1)

            # apply random erasing
            sample = random_erasing(sample.clone().detach())

            # revert min-max transformation
            sample = (sample * (this_max - this_min)) + this_min

        return sample, self.y[idx, ...], self.weights[idx, ...]


class Task5Model(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.bw2col = nn.Sequential(nn.BatchNorm2d(1))

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.mv2.classifier = nn.Sequential(nn.Linear(1280, 31))

    def forward(self, x):
        x = self.bw2col(x)
        x = torch.cat((x, x, x), 1)
        x = self.mv2(x)
        return x
