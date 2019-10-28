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
The technical report can read [here,](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Adapa_80.pdf) and the [workshop paper](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Adapa_83.pdf) is available on the [DCASE proceedings page](http://dcase.community/workshop2019/proceedings).

## License
Unless otherwise stated, the contents of this repository are shared under the [MIT License](LICENSE).

## Citing
```
@inproceedings{Adapa2019,
    author = "Adapa, Sainath",
    title = "Urban Sound Tagging using Convolutional Neural Networks",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019)",
    address = "New York University, NY, USA",
    month = "October",
    year = "2019",
    pages = "5--9",
    abstract = "In this paper, we propose a framework for environmental sound classification in a low-data context (less than 100 labeled examples per class). We show that using pre-trained image classification models along with usage of data augmentation techniques results in higher performance over alternative approaches. We applied this system to the task of Urban Sound Tagging, part of the DCASE 2019. The objective was to label different sources of noise from raw audio data. A modified form of MobileNetV2, a convolutional neural network (CNN) model was trained to classify both coarse and fine tags jointly. The proposed model uses log-scaled Mel-spectrogram as the representation format for the audio data. Mixup, Random erasing, scaling, and shifting are used as data augmentation techniques. A second model that uses scaled labels was built to account for human errors in the annotations. The proposed model achieved the first rank on the leaderboard with Micro-AUPRC values of 0.751 and 0.860 on fine and coarse tags, respectively."
}
```
