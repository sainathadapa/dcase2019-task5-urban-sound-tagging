import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor

from utils import prepare_data, AudioDataset, Task5Model


# Load and prepare data
with open('./data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

unknown_to_known = (
    pd.merge(metadata['taxonomy_df'].loc[lambda x: x.fine_id == 'X', ['fine', 'coarse']],
             metadata['taxonomy_df'].loc[lambda x: x.fine_id != 'X', ['fine', 'coarse']],
             on='coarse', how='inner')
    .drop(columns='coarse')
    .groupby('fine_x')['fine_y']
    .apply(lambda x: list(x)).to_dict())
known_labels = metadata['taxonomy_df'].loc[lambda x: x.fine_id != 'X'].fine.tolist()

train_df = pd.concat([metadata['coarse_train'], metadata['fine_train']], axis=1, sort=True)
valid_df = pd.concat([metadata['coarse_test'], metadata['fine_test']], axis=1, sort=True)

# manual correction for one data point
train_df.loc[(train_df.sum(axis=1) == 37).copy(), :] = 0
valid_df.loc[(valid_df.sum(axis=1) == 37).copy(), :] = 0

train_X, train_y, train_y_mask = prepare_data(train_df, unknown_to_known)
valid_X, valid_y, valid_y_mask = prepare_data(valid_df, unknown_to_known)

# Channel wise normalization
channel_means = train_X.reshape(-1, 128).mean(axis=0).reshape(1, 1, 1, -1)
channel_stds = train_X.reshape(-1, 128).std(axis=0).reshape(1, 1, 1, -1)
train_X = (train_X - channel_means) / channel_stds
valid_X = (valid_X - channel_means) / channel_stds
np.save('data/channel_means.npy', channel_means)
np.save('data/channel_stds.npy', channel_stds)

# Define the data augmentation transformations
albumentations_transform = Compose([
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
    GridDistortion(),
    ToTensor()
])

# Create the datasets and the dataloaders
train_dataset = AudioDataset(torch.Tensor(train_X),
                             torch.Tensor(train_y),
                             torch.Tensor(train_y_mask),
                             albumentations_transform)
valid_dataset = AudioDataset(torch.Tensor(valid_X),
                             torch.Tensor(valid_y),
                             torch.Tensor(valid_y_mask),
                             None)

val_loader = DataLoader(valid_dataset, 64, shuffle=False)
train_loader_1 = DataLoader(train_dataset, 64, shuffle=True)
train_loader_2 = DataLoader(train_dataset, 64, shuffle=True)

# Define the device to be used
cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)

# Instantiate the model
model = Task5Model(31).to(device)
print(model)

# Define optimizer, scheduler and loss criteria
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
criterion = nn.BCEWithLogitsLoss(reduction='none')

epochs = 100
train_loss_hist = []
valid_loss_hist = []
lowest_val_loss = np.inf
epochs_without_new_lowest = 0

for i in range(epochs):
    print('Epoch: ', i)

    this_epoch_train_loss = 0
    for i1, i2 in zip(train_loader_1, train_loader_2):

        # mixup the inputs ---------
        alpha = 1
        mixup_vals = np.random.beta(alpha, alpha, i1[0].shape[0])

        lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
        inputs = (lam * i1[0]) + ((1 - lam) * i2[0])

        lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1))
        labels = (lam * i1[1]) + ((1 - lam) * i2[1])
        masks = (lam * i1[2]) + ((1 - lam) * i2[2])
        # mixup ends ----------

        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            model = model.train()
            outputs = model(inputs)
            # calculate loss for each set of annotations
            loss_0 = criterion(outputs, labels[:, :, 0]) * masks[:, :, 0]
            loss_1 = criterion(outputs, labels[:, :, 1]) * masks[:, :, 1]
            loss_2 = criterion(outputs, labels[:, :, 2]) * masks[:, :, 2]
            loss = (loss_0.sum() + loss_1.sum() + loss_2.sum()) / masks.sum()
            loss.backward()
            optimizer.step()
            this_epoch_train_loss += loss.detach().cpu().numpy()

    this_epoch_valid_loss = 0
    for inputs, labels, masks in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            loss_0 = criterion(outputs, labels[:, :, 0]) * masks[:, :, 0]
            loss_1 = criterion(outputs, labels[:, :, 1]) * masks[:, :, 1]
            loss_2 = criterion(outputs, labels[:, :, 2]) * masks[:, :, 2]
            loss = (loss_0.sum() + loss_1.sum() + loss_2.sum()) / masks.sum()
            this_epoch_valid_loss += loss.detach().cpu().numpy()

    this_epoch_train_loss /= len(train_loader_1)
    this_epoch_valid_loss /= len(val_loader)

    train_loss_hist.append(this_epoch_train_loss)
    valid_loss_hist.append(this_epoch_valid_loss)

    if this_epoch_valid_loss < lowest_val_loss:
        lowest_val_loss = this_epoch_valid_loss
        torch.save(model.state_dict(), './data/model_system1')
        epochs_without_new_lowest = 0
    else:
        epochs_without_new_lowest += 1

    if epochs_without_new_lowest >= 25:
        break

    print(this_epoch_train_loss, this_epoch_valid_loss)

    scheduler.step(this_epoch_valid_loss)
