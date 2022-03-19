# CEM
> The official implementation for the paper *CEM: Commonsense-aware Empathetic Response Generation*.

<img src="https://img.shields.io/badge/Venue-AAAI--22-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red"> <img src="https://img.shields.io/badge/Last%20Updated-2022--03--19-2D333B" alt="update"/>

## Usage

### Dependencies

Install the required libraries (Python 3.8.5 | CUDA 10.2)

```sh
pip install -r requirements.txt 
```

Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

### Dataset

The preprocessed dataset is already provided as `/data/ED/dataset_preproc`. However, if you want to create the dataset yourself, delete this file, download the [COMET checkpoint](https://github.com/allenai/comet-atomic-2020) and place it in `/data/ED/Comet`. The preprocessed dataset would be generated after the training script.

### Training

```sh
python main.py --model [model_name] [--woDiv] [--woEMO] [--woCOG] [--cuda]
```

where model_name could be one of the following: **trs | multi-trs | moel | mime | empdg | cem**. In addition, the extra flags can be used for ablation studies.

## Testing

For reproducibility, download the trained [checkpoint](https://drive.google.com/file/d/1p_Qj5hBQE7e8ailIb5LbZu7NABmeet4k/view?usp=sharing),  put it in a folder named  `saved` and run the following:

```sh
python main.py --model cem --test --model_path save/CEM_19999_41.8034 [--cuda]
```

### Evaluation

Create a folder `results` and move the obtained results.txt for each model to this folder. Rename the files to the name of the model and run the following:

```sh
python src/scripts/evaluate.py 
```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{CEM2021,
      title={CEM: Commonsense-aware Empathetic Response Generation}, 
      author={Sahand Sabour, Chujie Zheng, Minlie Huang},
      journal={arXiv preprint arXiv:2109.05739},
      year={2021},
}
```
