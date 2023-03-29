# Single-cell resolution cross integration of multi-batch and multimodal omics data using DeepMerge

Multimodal single-cell omics technologies are revolutionising the field of molecular and cell biology, allowing joint profiling of multiple molecular features at single-cell resolution. Data generated from these technologies are characterised by multimodality and multi-batch, and their integration is essential for precise and holistic characterisation of cellular and biological systems. However, the lack of methods for simultaneous batch effect removal and data modality integration, termed “cross integration”, prevents discoveries that would be observable when the modalities of a single cell are analysed together. Here, we developed DeepMerge, a multi-task deep learning framework, for cross integration of multimodal single-cell omics data. Using a large collection of datasets, we demonstrate that DeepMerge enables versatile integration of various combinations of multi-batch and multimodal singlecell omics datasets that is unachievable by alternative methods and facilitates accurate separation of cell-type subpopulations and identification of molecular features underpinning cell identity. Together, we present a much-needed method for multimodal single-cell omics data analyses


<img width=100% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/main.png"/>


## Installation
DeepMerge is developed using PyTorch 1.9.1. We recommend using conda enviroment to install and run DeepMerge. We assume conda is installed. Note the following installation code snippets were tested on a Ubuntu system (v20.04) with NVIDIA GeForce 3090 GPU. The installation process needs about 5 minutes.

### Installation using provided environment
Step 1: Create and activate the conda environment for DeepMerge using our provided file
```
conda env create -f environment_deepmerge.yaml
conda activate environment_deepmerge
```

Step 2:
Otain DeepMerge by clonning the github repository:
```
git clone https://github.com/liuchunlei0430/DeepMerge.git
```


## Preparing intput for DeepMerge
DeepMerge’s main function takes raw count expression data (e.g., RNA, ADT, ATAC) in `.h5` format; also, it takes cell type labels and batch informations in `.csv` format. 

An example for creating .h5 file from expression matrix in the R environment is as below:
```
write_h5 <- function(exprs_list, h5file_list) {  
  for (i in seq_along(exprs_list)) {
    h5createFile(h5file_list[i])
    h5createGroup(h5file_list[i], "matrix")
    writeHDF5Array(t((exprs_list[[i]])), h5file_list[i], name = "matrix/data")
    h5write(rownames(exprs_list[[i]]), h5file_list[i], name = "matrix/features")
    h5write(colnames(exprs_list[[i]]), h5file_list[i], name = "matrix/barcodes")
  }  
}
write_h5(exprs_list = list(rna = train_rna, h5file_list = "/DeepMerge/data/Rama/rna.h5")
```

### Example dataset

As an example, the processed CITE-seq dataset by Ramaswamy et al. (GSM166489)[1] is provided for the example run, which is saved in `./DeepMerge/data/Rama/`. The data can be downloaded at [link](https://drive.google.com/drive/folders/1nYtSKP8b_ZBlykGwkY3hkjrTIHkc1v0B?usp=sharing).
Users can prepare the example dataset as input for DeepMerge or use their own datasets.
Training and testing on demo dataset will cost no more than 1 minute with GPU.

## Running DeepMerge with the example dataset
### Training the DeepMerge model (see Arguments section for more details).
```
cd DeepMerge
sh run.sh
```
or you can specific the parameters on terminal
```
cd DeepMerge
python main.py --lr 0.02 --epochs 10 --batch_size 256 --hidden_modality1 185 --hidden_modality2 30  --modality1_path "./data/Rama/rna.h5"  --modality2_path "./data/Rama/adt.h5"  --cty_path "./data/Rama/cty.csv" --batch_path "./data/Rama/batch.csv" --dataset "Rama" --modality1 "rna" --modality2 "adt"
```

### Argument
Training dataset information
+ `--modality1_path`: path to the first modality.
+ `--modality2_path`: path to the second modality.
+ `--modality3_path`: path to the third modality (can be null if there is no data provided). 
+ `--cty_path`: path to the labels of data.
+ `--batch_path`: path to the batch information of data.

Training and model config
+ `--batch_size`: Batch size.
+ `--epochs`: Number of epochs.
+ `--lr`: Learning rate.
+ `--z_dim`: Dimension of latent space.
+ `--hidden_modality1`: Hidden layer dimension of the first modality branch.
+ `--hidden_modality2`: Hidden layer dimension of the second modality branch.
+ `--hidden_modality3`: Hidden layer dimension of the third modality branch.

Other config
+ `--seed`: The random seed for training.
+ `--modality1`: The name of the first modality.
+ `--modality2`: The name of the second modality.
+ `--modality3`: The name of the third modality.

## Visualisation
The TSNE visualisation of original data are:

<img width=50% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/original.png"/>

By running the `./qc/plot_TSNE.Rmd`, we can obtain the TSNE of the batch corrected data:

<img width=58% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/batch_corrected.png"/>

## Instructions on how to run DeepMerge on your data
To adjust these analyses to your dataset, adjust the inputs to provide your data, cell type labels, and batch information. The data is a .h5 file with all data from one modality. If we have multiple modalities, we need to have multiple data files which have the same cell order. The cell type label and batch files are .csv files for all data.

## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).


## License

This project is covered under the Apache 2.0 License.
