
## ChromeGCN
#### Graph Neural Networks for DNA Sequence Classification

This repository contains a PyTorch implementation of ChromeGCN from [Graph Convolutional Networks for Epigenetic State Prediction Using Both Sequence and 3D Genome Data (Lanchantin and Qi 2019)](https://www.biorxiv.org/content/10.1101/840173v1)


```
@article{lanchantin2019graph,
  title={Graph Convolutional Networks for Epigenetic State Prediction Using Both Sequence and 3D Genome Data},
  author={Lanchantin, Jack and Qi, Yanjun},
  journal={BioRxiv},
  pages={840173},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```



### Get the data

Download the raw and processed data using the following command (13GB zipped, 90GB unzipped):
```bash
wget http://chromegcn.s3.amazonaws.com/processed_data.tar.gz
mkdir data/processed_data/
tar -xvf processed_data.tar.gz -C data/processed_data/
```

(optional) If you want to re-process the raw data below and follow the instructions in data/README.md
```bash
wget http://chromegcn.s3.amazonaws.com/data.tar.gz
mkdir data/orig_data/
tar -xvf data.tar.gz -C data/orig_data/
```


### Pretrain the Independent Window Model (CNN)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  -batch_size 64 -d_model 128 -epochs 100 -dropout 0.2  -lr 0.25 -window_model 'expecto' -optim 'sgd' -cell_type 'GM12878' -pretrain -shuffle_train -dataroot './data/processed_data/' -results_dir './results/'
```

### Save features from best epoch (use same flags)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -batch_size 64 -d_model 128 -epochs 100 -dropout 0.2  -lr 0.25 -window_model 'expecto' -optim 'sgd' -cell_type 'GM12878' -save_feats -dataroot './data/processed_data/' -results_dir './results/' 
```

### Train the 3D Genome Chromosome Model (GCN)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -batch_size 64 -d_model 128 -epochs 1000 -dropout 0.2  -window_model 'expecto' -chrome_model 'gcn' -optim 'sgd' -lr 0.25 -load_pretrained -lr2 0.25 -optim2 'sgd' -chrome_model 'gcn' -gate -gcn_layers 2 -adj_type 'hic' -hicnorm 'SQRTVC' -cell_type 'GM12878' -overwrite -hicsize 500000 -dataroot './data/processed_data/' -results_dir './results/'
```



