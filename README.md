# NeuroNER based system for reproducing paper results

In the paper *Customization Scenarios for De-identification of Clinical Notes*
by Hartman et al. from Google LLC, we used a text de-identification system
that's very similar to this code. This code is based on NeuroNER, an open source
program that performs named-entity recognition (NER). Website:
[neuroner.com](http://neuroner.com).

NOTICE: This is not an officially supported Google product.

## Notes

-   Embedding:
    [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)
-   Types: `AGE`, `CITY`, `DATE`, `EMAIL`, `ID`, `MEDICALRECORD`, `NAME`,
    `PHONE`, `STREET`, `ZIPCODE`
-   We have released a labeling of Physionet Gold Corpus following the I2B2-2014
    guidelines in order to facilitate fair comparison
    [deid-annotations](https://g.co/kaggle/deid-annotations).
-   We comprised three versions for each note in our train set to robustify our
    model to case changes: lower, upper and original.

## Utility scripts

-   **Model Training:**

    `run.py --train --dataset_text_folder=<dataset>
    --token_embedding_dimension=300 --output_folder=<something> --threads_tf=128
    --threads_prediction=10`

-   **Evaluate saved training models @epoch:**

    -   **Save model @epoch to `trained_model` directory:**

        `share_model.py`

    -   **Run evaluation of a model on a dataset:**

        `util.py --eval --pretrained_model_folder=<trained_model>
        --dataset_text_folder=<dataset> --rbias=0 --edim=300`

    -   **Evaluate binary/typed results in `results.json`:**

        `eval.py --metrics=<binary|token> --datasets=<eval-on-dataset-folder>`


Next is the rest of the original NeuroNER `README.md`.


## NeuroNER

[![Build Status](https://travis-ci.org/Franck-Dernoncourt/NeuroNER.svg?branch=master)](https://travis-ci.org/Franck-Dernoncourt/NeuroNER)

NeuroNER is a program that performs named-entity recognition (NER). Website: [neuroner.com](http://neuroner.com).

This page gives step-by-step instructions to install and use NeuroNER.


## Table of Contents

<!-- toc -->

- [Requirements](#requirements)
- [Installation](#installation)
- [Using NeuroNER](#using-neuroner)
  * [Adding a new dataset](#adding-a-new-dataset)
  * [Using a pretrained model](#using-a-pretrained-model)
  * [Sharing a pretrained model](#sharing-a-pretrained-model)
  * [Using TensorBoard](#using-tensorboard)
- [Citation](#citation)

<!-- tocstop -->

## Requirements

NeuroNER relies on Python 3, TensorFlow 1.0+, and optionally on BRAT:

- Python 3: NeuroNER does not work with Python 2.x. On Windows, it has to be Python 3.6 64-bit or later.
- TensorFlow is a library for machine learning. NeuroNER uses it for its NER engine, which is based on neural networks. Official website: [https://www.tensorflow.org](https://www.tensorflow.org)
- BRAT (optional) is a web-based annotation tool. It only needs to be installed if you wish to conveniently create annotations or view the predictions made by NeuroNER. Official website: [http://brat.nlplab.org](http://brat.nlplab.org)

## Installation


```
# Install build dependencies
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git
# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to .bashrc manually
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Reload config
source ~/.bashrc

# Install Python 3.7.12
pyenv install 3.7.12

# Create and activate environment
pyenv virtualenv 3.7.12 neuroner37
pyenv activate neuroner37


#install dependencies
pip install -r requirements.txt

#download and unzip embeddings
mkdir -p data/word_vectors
wget -P data/word_vectors http://neuroner.com/data/word_vectors/glove.6B.100d.zip
unzip data/word_vectors/glove.6B.100d.zip -d data/word_vectors/

#download spacy en model
python -m spacy download en

```
## Usage
1. Fetch pretrained models
```

neuroner --fetch_trained_model=conll_2003_en
neuroner --fetch_trained_model=i2b2_2014_glove_spacy_bioes
neuroner --fetch_trained_model=i2b2_2014_glove_stanford_bioes
neuroner --fetch_trained_model=mimic_glove_spacy_bioes
neuroner --fetch_trained_model=mimic_glove_stanford_bioes
```
2. Create/import datasets
NeuroNER requires datasets to be in BRAT format, structured into train/, valid/, and test/ folders, with matching .txt and .ann files.
To create a BRAT-formatted dataset from:
    The PhysioNet raw notes (orig.txt) not provided
    The Kaggle-provided annotation CSV (ann.csv) This is provided already
Use the following script:
 
```
python scripts/process_dataset.py \
  --text data/rawdataset/<nameoftxtfile>.txt \
  --annotations data/rawdataset/ann.csv \
  --output data/<name of folder you want> \
  --mappings

```
Arguments:

--text: path to the raw note file from PhysioNet. This is not provided

--annotations: path to the Kaggle CSV file. This is provided

--output: target output directory (e.g., split_output)

--mappings (optional): apply label mapping
If you want to map labels modify the dictionary in the script. It's default to
NAME         → NAME
LOCATION     → LOCATION
CITY         → CITY
STATE        → STATE
ID           → ID
ORGANIZATION → ORGANIZATION
PROFESSION   → PROFESSION
DATE         → DATE
AGE          → AGE
COUNTRY      → COUNTRY
HOSPITAL     → HOSPITAL
PHONE        → PHONE

Make sure the labels in your dataset matches what the pretrained model expects. 
You can find the expected labels by inspecting dataset/ inside the model folder.

3. Evaluating pretrained models on custom dataset
```
neuroner --train_model=False \
         --use_pretrained_model=True \
         --dataset_text_folder= data/<dataset name>\
         --pretrained_model_folder=trained_models/<model name> \
         --main_evaluation_mode=token

```
Make sure labels of dataset matches model, if not map the labels with the below command after modifying diciontary
input directory must have train val and test folders
```
python scripts/modify_labels.py \
  --input data/<original dataset split directory> \
  --output data/<output directory>
```

To see what the model actually predicted for each token look at the txt files in the output folder

4. Finetuning pretrained models on custom dataset
```
neuroner --train_model=True \
         --use_pretrained_model=True \
         --dataset_text_folder=data/<dataset name> \
         --pretrained_model_folder=trained_models/<model name>

```
Make sure labels match. To use this model, look inside the output directory and the folder that has the same name 
as the dataset. Copy the checkpoint file, dataset file, parameters file, and .data, .index, and .meta file of the 
epoch you want and move these all into a folder and put them in the trained_models folder. 
Make sure to rename the parameters file to parameters.ini and remove the epoch numbers. it should look like below
https://drive.google.com/file/d/1vRS6wXauP6Qv8Vfhu_tSiMIbqY6KwZzU/view?usp=sharing

To increase maximum epochs allowed without improvement add a --patience=<numberofepochs> arguement, default is set to 10




5. Training model from scratch on custom dataset
```
neuroner --train_model=True \
         --use_pretrained_model=False \
         --dataset_text_folder=data/<dataset name> \
         
```
To increase maximum epochs allowed without improvement add a --patience=<numberofepochs> arguement, default is set to 10

To use this model, look inside the output directory and the folder that has the same name 
as the dataset. Copy the checkpoint file, dataset file, parameters file, and .data, .index, and .meta file of the 
epoch you want and move these all into a folder and put them in the trained_models folder. 
Make sure to rename the parameters file to parameters.ini and remove the epoch numbers. it should look like below
https://drive.google.com/file/d/1vRS6wXauP6Qv8Vfhu_tSiMIbqY6KwZzU/view?usp=sharing

6. Analyze Model Output for particular label
   
   To see the true positives, false positives, and false negatives and recall,prec for a particular label
```
python scripts/AnalyzeLabel.py   
--file output/<name of output folder>/<name of txt file>   
--label <label>

```
look inside output folder and find the folder for the evaluation
name of txt file should either be 000_test.txt or 000_train.txt depending on what set you want to analyze
if you pass label as PATIENT, it will count both PATIENT and DOCTOR as true positives





# To use the CPU if you have installed tensorflow, or use the GPU if you have installed tensorflow-gpu:
neuroner

# To use the CPU only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES="" neuroner

# To use the GPU 1 only if you have installed tensorflow-gpu:
CUDA_VISIBLE_DEVICES=1 neuroner
```

If you wish to change any of NeuroNER parameters, you can modify the [`parameters.ini`](parameters.ini) configuration file in your working directory or specify it as an argument.

For example, to reduce the number of training epochs and not use any pre-trained token embeddings:

```
neuroner --maximum_number_of_epochs=2 --token_pretrained_embedding_filepath=""
```

To perform NER on some plain texts using a pre-trained model:

```
neuroner --train_model=False --use_pretrained_model=True --dataset_text_folder=./data/example_unannotated_texts --pretrained_model_folder=./trained_models/conll_2003_en
```

If a parameter is specified in both the [`parameters.ini`](parameters.ini) configuration file and as an argument, then the argument takes precedence (i.e., the parameter in [`parameters.ini`](parameters.ini) is ignored). You may specify a different configuration file with the `--parameters_filepath` command line argument. The command line arguments have no default value except for `--parameters_filepath`, which points to [`parameters.ini`](parameters.ini).

NeuroNER has 3 modes of operation:

- training mode (from scratch): the dataset folder must have train and valid sets. Test and deployment sets are optional.
- training mode (from pretrained model): the dataset folder must have train and valid sets. Test and deployment sets are optional.
- prediction mode (using pretrained model): the dataset folder must have either a test set or a deployment set.

### Adding a new dataset

A dataset may be provided in either CoNLL-2003 or BRAT format. The dataset files and folders should be organized and named as follows:

- Training set: `train.txt` file (CoNLL-2003 format) or `train` folder (BRAT format). It must contain labels.
- Validation set: `valid.txt` file (CoNLL-2003 format) or `valid` folder (BRAT format). It must contain labels.
- Test set: `test.txt` file (CoNLL-2003 format) or `test` folder (BRAT format). It must contain labels.
- Deployment set: `deploy.txt` file (CoNLL-2003 format) or `deploy` folder (BRAT format). It shouldn't contain any label (if it does, labels are ignored).

We provide several examples of datasets:

- [`data/conll2003/en`](data/conll2003/en): annotated dataset with the CoNLL-2003 format, containing 3 files (`train.txt`, `valid.txt` and  `test.txt`).
- [`data/example_unannotated_texts`](data/example_unannotated_texts): unannotated dataset with the BRAT format, containing 1 folder (`deploy/`). Note that the BRAT format with no annotation is the same as plain texts.

### Using a pretrained model

In order to use a pretrained model, the `pretrained_model_folder` parameter in the [`parameters.ini`](parameters.ini) configuration file must be set to the folder containing the pretrained model. The following parameters in the [`parameters.ini`](parameters.ini) configuration file must also be set to the same values as in the configuration file located in the specified `pretrained_model_folder`:

```
use_character_lstm
character_embedding_dimension
character_lstm_hidden_state_dimension
token_pretrained_embedding_filepath
token_embedding_dimension
token_lstm_hidden_state_dimension
use_crf
tagging_format
tokenizer
```

### Sharing a pretrained model

You are highly encouraged to share a model trained on their own datasets, so that other users can use the pretrained model on other datasets. We provide the [`neuroner/prepare_pretrained_model.py`](neuroner/prepare_pretrained_model.py) script to make it easy to prepare a pretrained model for sharing. In order to use the script, one only needs to specify the `output_folder_name`, `epoch_number`, and `model_name` parameters in the script.

By default, the only information about the dataset contained in the pretrained model is the list of tokens that appears in the dataset used for training and the corresponding embeddings learned from the dataset.

If you wish to share a pretrained model without providing any information about the dataset (including the list of tokens appearing in the dataset), you can do so by setting

```delete_token_mappings = True```

when running the script. In this case, it is highly recommended to use some external pre-trained token embeddings and freeze them while training the model to obtain high performance. This can be done by specifying the `token_pretrained_embedding_filepath` and setting

```freeze_token_embeddings = True```

in the [`parameters.ini`](parameters.ini) configuration file during training.

In order to share a pretrained model, please [submit a new issue](https://github.com/Franck-Dernoncourt/NeuroNER/issues/new) on the GitHub repository.

### Using TensorBoard

You may launch TensorBoard during or after the training phase. To do so, run in the terminal from the NeuroNER folder:
```
tensorboard --logdir=output
```

This starts a web server that is accessible at http://127.0.0.1:6006 from your web browser.

## Citation

If you use NeuroNER in your publications, please cite this [paper](https://arxiv.org/abs/1705.05487):

```
@article{2017neuroner,
  title={{NeuroNER}: an easy-to-use program for named-entity recognition based on neural networks},
  author={Dernoncourt, Franck and Lee, Ji Young and Szolovits, Peter},
  journal={Conference on Empirical Methods on Natural Language Processing (EMNLP)},
  year={2017}
}
```

The neural network architecture used in NeuroNER is described in this [article](https://arxiv.org/abs/1606.03475):

```
@article{2016deidentification,
  title={De-identification of Patient Notes with Recurrent Neural Networks},
  author={Dernoncourt, Franck and Lee, Ji Young and Uzuner, Ozlem and Szolovits, Peter},
  journal={Journal of the American Medical Informatics Association (JAMIA)},
  year={2016}
}
```
