# DCASE 2019 - Task 5 - Urban Sound Tagging

This repository contains the final solution that I used for the [DCASE 2019 - Task 5 - Urban Sound Tagging](http://dcase.community/challenge2019/task-urban-sound-tagging). The model achieved 1st position in prediction of both Coarse and Fine-level labels.

## Reproducing the results
**Prerequisites**:
- Linux based system
- Python >= 3.5
- NVidia GFX card with at least 8GB memory
- Cuda >= 10.0
- `virtualenv` package installed

**Replicating**:

Clone this repository. For a single command to replicate the entire solution, execute `make run_all` command while being in the repository directory. This command does the following steps sequentially:
- `make env`: Creates a virtual environment in the current directory
- `make reqs`: Installs python packages
- `make pytorch`: Installs PyTorch
- `make download`: Downloads the Task 5's data from Zenodo
- `make extract`: Extracts the zipped files
- `make parse`: Parses annotations
- `make logmel`: Computes and saves Log-Mel spectrograms for all the files
- `make train_s1`: Trains (system 1) model
- `make eval_s1`: Conducts local evaluation of the trained model (system 1)
- `make submit_s1`: Generates the submission file (system 1)
- `make train_s2`: Trains (system 2) model
- `make eval_s2`: Conducts local evaluation of the trained model (system 2)
- `make submit_s2`: Generates the submission file (system 2)

## Artifacts
The weights for both the models are available in the `releases` page.

## About the solution
The technical report can read [here](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Adapa_80.pdf)

## License
Unless otherwise stated, the contents of this repository are shared under the [MIT License](LICENSE).

## Citing
```
@misc{adapa2019urban,
    title={Urban Sound Tagging using Convolutional Neural Networks},
    author={Sainath Adapa},
    year={2019},
    eprint={1909.12699},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```
