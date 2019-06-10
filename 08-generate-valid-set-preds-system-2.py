import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from utils import Task5Model


with open('./data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

validate_files = list(set(metadata['coarse_test'].index.tolist()))

X = np.concatenate([
        np.expand_dims(np.load('./data/logmelspec/{}.npy'.format(x)).T[:635, :], axis=0)
        for x in validate_files])
X = X[:, None, :, :]

channel_means = np.load('./data/channel_means.npy')
channel_stds = np.load('./data/channel_stds.npy')
X = (X - channel_means) / channel_stds


class AudioDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx, ...]
        i = np.random.randint(sample.shape[1])
        sample = torch.cat([
                sample[:, i:, :],
                sample[:, :i, :]],
                dim=1)
        return sample


valid_dataset = AudioDataset(torch.Tensor(X))
valid_loader = DataLoader(valid_dataset, 64, shuffle=False)

cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)

model = Task5Model(37).to(device)

model.load_state_dict(torch.load('./data/model_system2'))

all_preds = []
for _ in range(10):
    preds = []
    for inputs in valid_loader:
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                preds.append(outputs.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    preds = (1 / (1 + np.exp(-preds)))
    all_preds.append(preds)
tmp = all_preds[0]
for x in all_preds[1:]:
    tmp += x
tmp = tmp / 10
preds = tmp

output_df = pd.DataFrame(
    preds, columns=[
        '1-1_small-sounding-engine', '1-2_medium-sounding-engine',
        '1-3_large-sounding-engine', '1-X_engine-of-uncertain-size',
        '1_engine', '2-1_rock-drill', '2-2_jackhammer',
        '2-3_hoe-ram', '2-4_pile-driver', '2-X_other-unknown-impact-machinery',
        '2_machinery-impact', '3-1_non-machinery-impact', '3_non-machinery-impact',
        '4-1_chainsaw', '4-2_small-medium-rotating-saw', '4-3_large-rotating-saw',
        '4-X_other-unknown-powered-saw', '4_powered-saw', '5-1_car-horn',
        '5-2_car-alarm', '5-3_siren', '5-4_reverse-beeper',
        '5-X_other-unknown-alert-signal', '5_alert-signal', '6-1_stationary-music',
        '6-2_mobile-music', '6-3_ice-cream-truck', '6-X_music-from-uncertain-source',
        '6_music', '7-1_person-or-small-group-talking',
        '7-2_person-or-small-group-shouting', '7-3_large-crowd', '7-4_amplified-speech',
        '7-X_other-unknown-human-voice', '7_human-voice', '8-1_dog-barking-whining',
        '8_dog'])
output_df['audio_filename'] = pd.Series(
    validate_files,
    index=output_df.index)

output_df.to_csv('data/valid-set-preds-system-2.csv', index=False)
