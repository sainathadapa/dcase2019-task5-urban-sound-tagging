PIP := .env/bin/pip
PYTHON := .env/bin/python

env:
	echo "Create virtual environment..."
	virtualenv .env -p python3

rmenv:
	echo "Remove virtual environment..."
	rm -rf .env

reqs:
	echo "Install python packages...(Excluding PyTorch)"
	$(PIP) install -r requirements.txt

pytorch:
	echo "Install PyTorch..."
	$(PIP) install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
	$(PIP) install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

download:
	echo "Downloading the data..."
	mkdir -p data
	wget -O "data/dcase-ust-taxonomy.yaml" "https://zenodo.org/record/3233082/files/dcase-ust-taxonomy.yaml?download=1"
	wget -O "data/audio-eval.tar.gz" "https://zenodo.org/record/3233082/files/audio-eval.tar.gz?download=1"
	wget -O "data/audio-dev.tar.gz" "https://zenodo.org/record/3233082/files/audio-dev.tar.gz?download=1"
	wget -O "data/annotations-dev.csv" "https://zenodo.org/record/3233082/files/annotations-dev.csv?download=1"

extract:
	echo "Extracting the files..."
	tar -xvzf data/audio-dev.tar.gz -C data/
	tar -xvzf data/audio-eval.tar.gz -C data/

parse:
	echo "Parse annotations data..."
	$(PYTHON) 01-parse-annotations.py

logmel:
	echo "Computing log-mel spectrograms..."
	$(PYTHON) 02-compute-log-mel.py
	
train_s1:
	echo "Training the model...(system 1)"
	$(PYTHON) 03-train-system-1.py

eval_s1:
	echo "Local evaluation on validation set...(system 1)"
	$(PYTHON) 04-generate-valid-set-preds-system-1.py
	$(PYTHON) baseline_code/evaluate_predictions.py data/valid-set-preds-system-1.csv data/annotations-dev.csv data/dcase-ust-taxonomy.yaml

submit_s1:
	echo "Generate submission...(system 1)"
	$(PYTHON) 09-generate-submission-system-1.py

train_s2:
	echo "Training the model...(system 2)"
	$(PYTHON) 06-scale-annos-system-2.py
	$(PYTHON) 07-train-system-2.py

eval_s2:
	echo "Local evaluation on validation set...(system 2)"
	$(PYTHON) 08-generate-valid-set-preds-system-2.py
	$(PYTHON) baseline_code/evaluate_predictions.py data/valid-set-preds-system-2.csv data/annotations-dev.csv data/dcase-ust-taxonomy.yaml

submit_s2:
	echo "Generate submission...(system 2)"
	$(PYTHON) 10-generate-submission-system-2.py

install: env reqs pytorch
	:

prep: download extract parse logmel
	:

system1: train_s1 eval_s1 submit_s1
	:

system2: train_s2 eval_s2 submit_s2
	:

run_all: install prep system1 system2
	:

