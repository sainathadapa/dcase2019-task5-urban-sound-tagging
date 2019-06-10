import argparse
import gzip
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from vggish import vggish_input
from vggish import vggish_postprocess
from vggish import vggish_slim


def make_extract_vggish_embedding(frame_duration, hop_duration, input_op_name='vggish/input_features',
                                  output_op_name='vggish/embedding', embedding_size=128, resources_dir=None):
    """
    Creates a coroutine generator for extracting and saving VGGish embeddings

    Parameters
    ----------
    frame_duration
    hop_duration
    input_op_name
    output_op_name
    embedding_size
    resources_dir

    Returns
    -------
    coroutine

    """
    params = {
        'frame_win_sec': frame_duration,
        'frame_hop_sec': hop_duration,
        'embedding_size': embedding_size
    }

    if not resources_dir:
        resources_dir = os.path.join(os.path.dirname(__file__), 'vggish/resources')

    pca_params_path = os.path.join(resources_dir, 'vggish_pca_params.npz')
    model_path = os.path.join(resources_dir, 'vggish_model.ckpt')

    try:
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False, **params)
            vggish_slim.load_vggish_slim_checkpoint(sess, model_path, **params)

            while True:
                # We use a coroutine to more easily keep open the Tensorflow contexts
                # without having to constantly reload the model
                audio_path, output_path = (yield)

                if os.path.exists(output_path):
                    continue

                try:
                    examples_batch = vggish_input.wavfile_to_examples(audio_path, **params)
                except ValueError:
                    print("Error opening {}. Skipping...".format(audio_path))
                    continue

                # Prepare a postprocessor to munge the model embeddings.
                pproc = vggish_postprocess.Postprocessor(pca_params_path, **params)

                input_tensor_name = input_op_name + ':0'
                output_tensor_name = output_op_name + ':0'

                features_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
                embedding_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})

                emb = pproc.postprocess(embedding_batch, **params).astype(np.float32)

                with gzip.open(output_path, 'wb') as f:
                    emb.dump(f)

    except GeneratorExit:
        pass


def extract_embeddings_vggish(annotation_path, dataset_dir, output_dir,
                              vggish_resource_dir, frame_duration=0.96,
                              hop_duration=0.96, progress=True,
                              vggish_embedding_size=128):
    """
    Extract embeddings for files annotated in the SONYC annotation file and save them to disk.

    Parameters
    ----------
    annotation_path
    dataset_dir
    output_dir
    vggish_resource_dir
    frame_duration
    hop_duration
    progress
    vggish_embedding_size

    Returns
    -------

    """

    print("* Loading annotations.")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')

    extract_vggish_embedding = make_extract_vggish_embedding(frame_duration, hop_duration,
        input_op_name='vggish/input_features', output_op_name='vggish/embedding',
        resources_dir=vggish_resource_dir, embedding_size=vggish_embedding_size)
    # Start coroutine
    next(extract_vggish_embedding)

    out_dir = os.path.join(output_dir, 'vggish')
    os.makedirs(out_dir, exist_ok=True)

    df = annotation_data[['split', 'audio_filename']].drop_duplicates()
    row_iter = df.iterrows()

    if progress:
        row_iter = tqdm(row_iter, total=len(df))

    print("* Extracting embeddings.")
    for _, row in row_iter:
        filename = row['audio_filename']
        split_str = row['split']
        audio_path = os.path.join(dataset_dir, split_str, filename)
        emb_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.npy.gz')
        extract_vggish_embedding.send((audio_path, emb_path))

    extract_vggish_embedding.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("vggish_resource_dir")

    parser.add_argument("--vggish_embedding_size", type=int, default=128)

    parser.add_argument("--frame_duration", type=float, default=0.96)
    parser.add_argument("--hop_duration", type=float, default=0.96)
    parser.add_argument("--progress", action="store_const", const=True, default=False)

    args = parser.parse_args()

    extract_embeddings_vggish(annotation_path=args.annotation_path,
                              dataset_dir=args.dataset_dir,
                              output_dir=args.output_dir,
                              vggish_resource_dir=args.vggish_resource_dir,
                              vggish_embedding_size=args.vggish_embedding_size,
                              frame_duration=args.frame_duration,
                              hop_duration=args.hop_duration,
                              progress=args.progress)
