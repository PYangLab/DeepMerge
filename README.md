# DeepMerge: a unified framework for diagonal integration of multi-batch multimodal single-cell omics data

DeepMerge is a method for batch correcting and integrating multimodal single-cell omics data using a multi-task deep learning framework. DeepMerge not only performs batch correction and integration across data modalities but also generates normalised and corrected data matrices that can be readily utilised for downstream analyses such as identifying differentially expressed genes, ADTs, and/or cis-regulatory elements (CREs) from multiple modalities of the batch corrected and integrated dataset. By applying DeepMerge to a large collection of datasets generated from various biotechnological platforms, we demonstrate its utility for integrative analyses of multi-batch multimodal single-cell omics datasets.


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

As an example, the processed CITE-seq dataset by RAMA et al. (GSM166489)[1] is provided for the example run, which is saved in `./DeepMerge/data/Rama/`. The data can be downloaded at [link](https://www.dropbox.com/scl/fo/e5cdogwhj6k8stjmo9s7w/h?dl=0&rlkey=sxuusr69cco4jmzqk7vmwopx).
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

## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).


## License

This project is covered under the Apache 2.0 License.
