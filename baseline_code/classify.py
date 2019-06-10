import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml

from sklearn.preprocessing import StandardScaler


## HELPERS

def load_embeddings(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory

    Parameters
    ----------
    file_list
    emb_dir

    Returns
    -------
    embeddings
    ignore_idxs

    """
    embeddings = []
    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npy.gz')
        with gzip.open(emb_path, 'rb') as f:
            embeddings.append(np.load(f))

    return embeddings


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data

    Returns
    -------
    train_idxs
    valid_idxs

    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []
    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        else:
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels

    Parameters
    ----------
    annotation_data
    labels

    Returns
    -------
    target_list

    """
    target_list = []
    file_list = annotation_data['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    # If we have a validated annotation, just use that
                    count = row[label + '_presence']
                    break
                else:
                    count += row[label + '_presence']

            if count > 0:
                target.append(1.0)
            else:
                target.append(0.0)

        target_list.append(target)

    return np.array(target_list)


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Courtesy of https://stackoverflow.com/a/42797620

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


## MODEL CONSTRUCTION


def construct_mlp_framewise(emb_size, num_classes, hidden_layer_size=128,
                            num_hidden_layers=0, l2_reg=1e-5):
    """
    Construct a 2-hidden-layer MLP model for framewise processing

    Parameters
    ----------
    emb_size
    num_classes
    hidden_layer_size
    num_hidden_layers
    l2_reg

    Returns
    -------
    model

    """
    # Input layer
    inp = Input(shape=(emb_size,), dtype='float32', name='input')
    y = inp

    # Add hidden layers
    for idx in range(num_hidden_layers):
        y = Dense(hidden_layer_size, activation='relu',
                  kernel_regularizer=regularizers.l2(l2_reg),
                  name='dense{}'.format(idx+1))(y)

    # Output layer
    y = Dense(num_classes, activation='sigmoid',
              kernel_regularizer=regularizers.l2(l2_reg), name='output')(y)

    m = Model(inputs=inp, outputs=y)
    m.name = 'urban_sound_classifier'

    return m


## DATA PREPARATION

def prepare_framewise_data(train_file_idxs, test_file_idxs, embeddings,
                           target_list, standardize=True):
    """
    Prepare inputs and targets for framewise training using training and evaluation indices.

    Parameters
    ----------
    train_file_idxs
    test_file_idxs
    embeddings
    target_list
    standardize

    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler

    """

    X_train = []
    y_train = []
    for idx in train_file_idxs:
        X_ = list(embeddings[idx])
        X_train += X_
        for _ in range(len(X_)):
            y_train.append(target_list[idx])

    train_idxs = np.random.permutation(len(X_train))

    X_train = np.array(X_train)[train_idxs]
    y_train = np.array(y_train)[train_idxs]

    X_valid = []
    y_valid = []
    for idx in test_file_idxs:
        X_ = list(embeddings[idx])
        X_valid += X_
        for _ in range(len(X_)):
            y_valid.append(target_list[idx])

    test_idxs = np.random.permutation(len(X_valid))
    X_valid = np.array(X_valid)[test_idxs]
    y_valid = np.array(y_valid)[test_idxs]

    # standardize
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
    else:
        scaler = None

    return X_train, y_train, X_valid, y_valid, scaler


## GENERIC MODEL TRAINING


def train_model(model, X_train, y_train, X_valid, y_valid, output_dir,
                loss=None, batch_size=64, num_epochs=100, patience=20,
                learning_rate=1e-4):
    """
    Train a model with the given data.

    Parameters
    ----------
    model
    X_train
    y_train
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate

    Returns
    -------
    history

    """

    if loss is None:
        loss = 'binary_crossentropy'
    # TODO: Update for our modified accuracy metric
    metrics = []
    #set_random_seed(random_state)

    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')

    cb.append(keras.callbacks.ModelCheckpoint(model_weight_file,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor='val_loss'))
    # early stopping
    cb.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=patience))

    # monitor losses
    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    # Fit model
    model.compile(Adam(lr=learning_rate), loss=loss, metrics=metrics)
    history = model.fit(
        x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
        validation_data=(X_valid, y_valid), callbacks=cb, verbose=2)

    return history


## MODEL TRAINING

def train_framewise(annotation_path, taxonomy_path, emb_dir, output_dir, exp_id,
                    label_mode="fine", batch_size=64, num_epochs=100,
                    patience=20, learning_rate=1e-4, hidden_layer_size=128,
                    num_hidden_layers=0, l2_reg=1e-5, standardize=True,
                    timestamp=None):
    """
    Train and evaluate a framewise MLP model.

    Parameters
    ----------
    dataset_dir
    emb_dir
    output_dir
    exp_id
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    timestamp

    Returns
    -------

    """

    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data['audio_filename'].unique().tolist()

    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                            if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    print("* Preparing training data.")

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, test_file_idxs = get_subset_split(annotation_data)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))

    num_classes = len(labels)

    embeddings = load_embeddings(file_list, emb_dir)

    X_train, y_train, X_valid, y_valid, scaler \
        = prepare_framewise_data(train_file_idxs, test_file_idxs, embeddings,
                                 target_list, standardize=standardize)

    _, emb_size = X_train.shape

    model = construct_mlp_framewise(emb_size, num_classes,
                                    hidden_layer_size=hidden_layer_size,
                                    num_hidden_layers=num_hidden_layers,
                                    l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
        loss_func = masked_loss
    else:
        loss_func = None

    print("* Training model.")
    history = train_model(model, X_train, y_train, X_valid, y_valid,
                          results_dir, loss=loss_func, batch_size=batch_size,
                          num_epochs=num_epochs, patience=patience,
                          learning_rate=learning_rate)

    print("* Saving model predictions.")
    results = {}
    results['train'] = predict_framewise(embeddings, train_file_idxs, model,
                                         scaler=scaler)
    results['test'] = predict_framewise(embeddings, test_file_idxs, model,
                                        scaler=scaler)
    results['train_history'] = history.history

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    for aggregation_type, y_pred in results['test'].items():
        generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                             aggregation_type, label_mode, taxonomy)


## MODEL EVALUATION

def predict_framewise(embeddings, test_file_idxs, model, scaler=None):
    """
    Evaluate the output of a framewise classification model.

    Parameters
    ----------
    embeddings
    test_file_idxs
    model
    scaler

    Returns
    -------
    results
    """
    y_pred_max = []
    y_pred_mean = []
    y_pred_softmax = []

    for idx in test_file_idxs:
        if scaler is None:
            X_ = np.array(embeddings[idx])
        else:
            X_ = np.array(scaler.transform(embeddings[idx]))
        pred_frame = model.predict(X_)

        y_pred_max.append(pred_frame.max(axis=0).tolist())
        y_pred_mean.append(pred_frame.mean(axis=0).tolist())
        y_pred_softmax.append(((softmax(pred_frame, axis=0) * pred_frame).sum(axis=0)).tolist())


    results = {
        'max': y_pred_max,
        'mean': y_pred_mean,
        'softmax': y_pred_softmax
    }

    return results


def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type, label_mode, taxonomy):
    """
    Write the output file containing model predictions

    Parameters
    ----------
    y_pred
    test_file_idxs
    results_dir
    file_list
    aggregation_type
    label_mode
    taxonomy

    Returns
    -------

    """
    output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]


    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')

    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_framewise(args.annotation_path,
                    args.taxonomy_path,
                    args.emb_dir,
                    args.output_dir,
                    args.exp_id,
                    label_mode=args.label_mode,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    hidden_layer_size=args.hidden_layer_size,
                    num_hidden_layers=args.num_hidden_layers,
                    l2_reg=args.l2_reg,
                    standardize=(not args.no_standardize),
                    timestamp=timestamp)
