import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

with open('./data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

valid_zen = pd.concat([
    metadata['coarse_test'],
    metadata['fine_test']
], axis=1, sort=True)

train_zen = pd.concat([
    metadata['coarse_train'],
    metadata['fine_train']
], axis=1, sort=True)

train_zen.loc[(train_zen.sum(axis=1) == 37).copy(), :] = 0
valid_zen.loc[(valid_zen.sum(axis=1) == 37).copy(), :] = 0

annotation_data = pd.read_csv('./data/annotations-dev.csv', low_memory=False).sort_values('audio_filename')
valid_gt = annotation_data.loc[lambda x: x.annotator_id == 0]

valid_gt = (valid_gt
            .set_index('audio_filename')
            .rename(columns=lambda x: x[:-9])
            .loc[:, train_zen.columns.tolist()])

valid_zen_long = (valid_zen.reset_index().groupby('audio_filename').sum().reset_index()
                  .melt(id_vars='audio_filename').rename(columns={'value': 'zn'}))
valid_gt_long = (valid_gt.reset_index().rename(columns={'index': 'audio_filename'})
                 .groupby('audio_filename').sum().reset_index()
                 .melt(id_vars='audio_filename').rename(columns={'value': 'gt'}))
valid_merged = pd.merge(valid_zen_long, valid_gt_long, on=['audio_filename', 'variable'], how='inner')

lr_model = LogisticRegression(solver='liblinear')

lr_model.fit(
    pd.get_dummies(valid_merged.loc[:, ['variable', 'zn']]).values,
    valid_merged['gt'].values)

train_long = (train_zen.reset_index().groupby('audio_filename').sum().reset_index()
              .melt(id_vars='audio_filename').rename(columns={'value': 'zn'}))

train_long['pred'] = pd.Series(lr_model.predict_proba(
    pd.get_dummies(train_long.loc[:, ['variable', 'zn']]).values
)[:, 1], index=train_long.index)

valid_zen_long['pred'] = pd.Series(lr_model.predict_proba(
    pd.get_dummies(valid_zen_long.loc[:, ['variable', 'zn']]).values
)[:, 1], index=valid_zen_long.index)

train_to_save = train_long.drop(columns='zn').pivot('audio_filename', 'variable', 'pred')
valid_to_save = valid_zen_long.drop(columns='zn').pivot('audio_filename', 'variable', 'pred')

train_to_save.to_pickle('./data/train_scaled.pkl')
valid_to_save.to_pickle('./data/valid_scaled.pkl')
