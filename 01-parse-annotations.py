import pickle
import yaml
import pandas as pd

annotation_data = pd.read_csv('./data/annotations-dev.csv', low_memory=False).sort_values('audio_filename')

with open('./data/dcase-ust-taxonomy.yaml', 'r') as f:
    taxonomy = yaml.load(f, Loader=yaml.Loader)

file_list = annotation_data['audio_filename'].unique().tolist()

full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                           for coarse_id, fine_dict in taxonomy['fine'].items()
                           for fine_id, fine_label in fine_dict.items()]

fine_target_labels = [x
                      for x in full_fine_target_labels
                      if x.split('_')[0].split('-')[1] != 'X']

coarse_target_labels = ["_".join([str(k), v])
                        for k, v in taxonomy['coarse'].items()]

tmp1 = pd.DataFrame.from_records([
    (key1, key2, val2)
    for key1, val1 in taxonomy['fine'].items()
    for key2, val2 in val1.items()
], columns=['coarse_id', 'fine_id', 'fine'])
tmp2 = pd.DataFrame.from_records([
    (key, val)
    for key, val in taxonomy['coarse'].items()
], columns=['coarse_id', 'coarse'])
taxonomy_df = pd.merge(tmp1, tmp2, on='coarse_id', how='inner')
taxonomy_df = taxonomy_df.astype('str')
taxonomy_df['coarse'] = taxonomy_df.coarse_id + '_' + taxonomy_df.coarse
taxonomy_df['fine'] = taxonomy_df.coarse_id + '-' + taxonomy_df.fine_id + '_' + taxonomy_df.fine

train_data = (annotation_data.copy()
              .loc[lambda x: x.split == 'train']
              .set_index('audio_filename')
              .filter(regex='.*_presence$', axis='columns')
              .rename(columns=lambda x: x[:-9])
              .astype('int64'))
coarse_train = train_data.loc[:, sorted(taxonomy_df.coarse.unique())]
fine_train = train_data.loc[:, sorted(taxonomy_df.fine.tolist())]

valid_data = (annotation_data.copy()
              .loc[lambda x: x.split == 'validate']
              .loc[lambda x: x.annotator_id > 0]
              .set_index('audio_filename')
              .filter(regex='.*_presence$', axis='columns')
              .rename(columns=lambda x: x[:-9])
              .astype('int64'))
coarse_valid = valid_data.loc[:, sorted(taxonomy_df.coarse.unique())]
fine_valid = valid_data.loc[:, sorted(taxonomy_df.fine.tolist())]

coarse_train.sort_index(inplace=True)
coarse_valid.sort_index(inplace=True)
fine_train.sort_index(inplace=True)
fine_valid.sort_index(inplace=True)

with open('./data/metadata.pkl', 'wb') as f:
    pickle.dump({
        'coarse_train': coarse_train,
        'coarse_test': coarse_valid,
        'fine_train': fine_train,
        'fine_test': fine_valid,
        'taxonomy_df': taxonomy_df
    }, f)
