import os
import numpy as np
import pandas as pd
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

from utils import Task5Model


eval_files = glob('./data/audio-eval/*.wav')
eval_files = [os.path.basename(x) for x in eval_files]

X = np.concatenate([
        np.expand_dims(np.load('./data/logmelspec-eval/{}.npy'.format(x)).T[:635, :], axis=0)
        for x in eval_files])
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


dataset = AudioDataset(torch.Tensor(X))
loader = DataLoader(dataset, 64, shuffle=False)

cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)

model = Task5Model(31).to(device)
model.load_state_dict(torch.load('./data/model_system1'))

all_preds = []
for _ in range(10):
    preds = []
    for inputs in loader:
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
        '1_engine', '2_machinery-impact', '3_non-machinery-impact',
        '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog',
        '1-1_small-sounding-engine', '1-2_medium-sounding-engine',
        '1-3_large-sounding-engine', '2-1_rock-drill', '2-2_jackhammer',
        '2-3_hoe-ram', '2-4_pile-driver', '3-1_non-machinery-impact',
        '4-1_chainsaw', '4-2_small-medium-rotating-saw',
        '4-3_large-rotating-saw', '5-1_car-horn', '5-2_car-alarm', '5-3_siren',
        '5-4_reverse-beeper', '6-1_stationary-music', '6-2_mobile-music',
        '6-3_ice-cream-truck', '7-1_person-or-small-group-talking',
        '7-2_person-or-small-group-shouting', '7-3_large-crowd',
        '7-4_amplified-speech', '8-1_dog-barking-whining'])
output_df['audio_filename'] = pd.Series(eval_files, index=output_df.index)

for x in [
        '1-X_engine-of-uncertain-size', '2-X_other-unknown-impact-machinery',
        '4-X_other-unknown-powered-saw', '5-X_other-unknown-alert-signal',
        '6-X_music-from-uncertain-source', '7-X_other-unknown-human-voice']:
    output_df[x] = 0

cols_in_order = [
    "audio_filename", "1-1_small-sounding-engine",
    "1-2_medium-sounding-engine", "1-3_large-sounding-engine",
    "1-X_engine-of-uncertain-size", "2-1_rock-drill",
    "2-2_jackhammer", "2-3_hoe-ram", "2-4_pile-driver",
    "2-X_other-unknown-impact-machinery", "3-1_non-machinery-impact",
    "4-1_chainsaw", "4-2_small-medium-rotating-saw",
    "4-3_large-rotating-saw", "4-X_other-unknown-powered-saw",
    "5-1_car-horn", "5-2_car-alarm", "5-3_siren", "5-4_reverse-beeper",
    "5-X_other-unknown-alert-signal", "6-1_stationary-music",
    "6-2_mobile-music", "6-3_ice-cream-truck",
    "6-X_music-from-uncertain-source", "7-1_person-or-small-group-talking",
    "7-2_person-or-small-group-shouting", "7-3_large-crowd",
    "7-4_amplified-speech", "7-X_other-unknown-human-voice",
    "8-1_dog-barking-whining", "1_engine", "2_machinery-impact",
    "3_non-machinery-impact", "4_powered-saw", "5_alert-signal",
    "6_music", "7_human-voice", "8_dog"]
output_df = output_df.loc[:, cols_in_order]

output_df.to_csv('data/submission-system-1.csv', index=False)
