# SciCo-Radar: Scientific Cross-Document Coreference and Hierarchy with Definition-Augmented Relational Reasoning

We present SciCo-Radar, methods to detect cross-document coreference and hierarchy in scientific papers by augmenting original inputs from papers with context-sensitive generated definitions and relational reasoning.

* Trained model on HF 
* Definitions files - https://drive.google.com/drive/folders/1SNM4hLR2sfXzViNpguTWqfum9qIk8sjV?usp=sharing
* Paper on arxiv

![Alt text](scico_def_main_fig_sketch.png?raw=true "Title")

## Baseline
To run and evaluate the baseline model, please follow the instructions in the original [SciCo model](https://github.com/ariecattan/SciCo/tree/main) repository. Make sure you run this project on a separate environment, as the original SciCo repo has many dependencies that might conflict with this one.

# Steps to run our model

Install all dependencies using `pip install -r requirements.txt` (we recommend using conda).

## 1. Data

First create a folder named **data** in the root directory.

### SciCo dataset

To download the raw data, click [here](https://nlp.biu.ac.il/~ariecattan/scico/data.tar).

Each file (train, dev, test) is in the `jsonl` format where each row corresponds a topic.
See below the description of the fields in each topic. After Downloading the data, place it in the **data** folder in the root directory.


You can also load SciCo directly from [huggingface.co/datasets/allenai/scico](https://huggingface.co/datasets/allenai/scico) as follows:

```python
from datasets import load_dataset
scico = load_dataset("allenai/scico")
```

### Definitions dataset

download the Singleton and Relational definitions datasets from [here](https://drive.google.com/drive/folders/1SNM4hLR2sfXzViNpguTWqfum9qIk8sjV?usp=sharing) and place them in the **data** folder in the root directory.

Create Definitions: coming soon

## 2. Training

To fine-tune the model, run the following command:

```python
python train_llm_classification.py --output_dir your-output-dir
```
or use accelerate launch to train on multiple GPUs

## Model

Our unified model is available on https://huggingface.co/allenai/longformer-scico.
We provide the following code as an example to set the global attention on the special tokens: `<s>`, `<m>` and `</m>`.


**Note:** There is a slight difference between this model and the original model presented in the [paper](https://openreview.net/forum?id=OFLbgUP04nC). 
The original model includes a single linear layer on top of the `<s>` token (equivalent to `[CLS]`) 
while this model includes a two-layers MLP to be in line with `LongformerForSequenceClassification`.   
You can download the original model as follows:
```python
curl -L -o model.tar https://www.dropbox.com/s/cpcnpov4liwuyd4/model.tar?dl=0
tar -xvf model.tar 
rm model.tar 
```


## Inference
To run inference on the model, execute the following command:

```python
python predict_llm_classification.py --adapter your-saved-model-path --output_dir your-output-dir 
```
or use accelerate launch to train on multiple GPUs

add --use_relational_def True to use the relational definitions

**Note:** The provided relational definitions were generated specifically with the model stated in the paper so scores can vary with other models.

Set save_path (where you want to save the results) and scores_path (where you saved the inference scores) in the config path in `configs/multiclass.yaml`,
then run the following script: 

```
python predict.py --config configs/multiclass.yaml 
```


### Evaluation 

After we finish the inference process it produces a `jsonl` file with the fields `tokens`, `mentions`, `relations` and `id`.
Models are evaluated using the usual coreference metrics using the [coval](https://github.com/ns-moosavi/coval/) script,
 hierarchy (recall, precision and F1), and directed path ratio. 

```
python evaluation.py [gold_jsonl_path] [sys_jsonl_path] options
```

If you want to evaluate only on the hard topics (based on levenshtein performance, see Section 4.5), 
you can set the `options` to be `hard_10`, `hard_20` or `curated`.
