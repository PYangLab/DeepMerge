# Single-cell resolution cross integration of multi-batch and multimodal omics data using DeepMerge

Multimodal single-cell omics technologies are revolutionising the field of molecular and cell biology, allowing joint profiling of multiple molecular features at single-cell resolution. Data generated from these technologies are characterised by multimodality and multi-batch, and their integration is essential for precise and holistic characterisation of cellular and biological systems. However, the lack of methods for simultaneous batch effect removal and data modality integration, termed “cross integration”, prevents discoveries that would be observable when the modalities of a single cell are analysed together. Here, we developed DeepMerge, a multi-task deep learning framework, for cross integration of multimodal single-cell omics data. Using a large collection of datasets, we demonstrate that DeepMerge enables versatile integration of various combinations of multi-batch and multimodal singlecell omics datasets that is unachievable by alternative methods and facilitates accurate separation of cell-type subpopulations and identification of molecular features underpinning cell identity. Together, we present a much-needed method for multimodal single-cell omics data analyses


<img width=100% src="https://github.com/PYangLab/DeepMerge/blob/main/img/main.png"/>


## Installation
DeepMerge is developed using PyTorch 1.9.1, and we recommend using a conda environment for its installation and execution. We assume that conda is already installed on your system. You can either use the provided environment or set up the environment yourself based on your hardware configurations. Please note that the following installation code snippets have been tested on an Ubuntu system (v20.04) with an NVIDIA GeForce 3090 GPU. The installation process should take approximately 5 minutes.

### Installation using provided environment
Step 1:
Otain DeepMerge by clonning the github repository:
```
git clone https://github.com/PYangLab/DeepMerge.git
cd DeepMerge
```

Step 2: Create and activate the conda environment for DeepMerge using our provided file
```
conda env create -f environment_deepmerge.yaml
conda activate environment_deepmerge
```

### Installation by youself
Step 1:
Otain DeepMerge by clonning the github repository:
```
git clone https://github.com/PYangLab/DeepMerge.git
cd DeepMerge
```

Step 2:
Create and activate the conda environment for DeepMerge
```
conda create -n environment_deepmerge python=3.7
conda activate environment_deepmerge
```

Step 3:
Check the environment including GPU settings and the highest CUDA version allowed by the GPU.
```
nvidia-smi
```

Step 4:
Install pytorch and cuda version based on your GPU settings according to [PyTorch](https://pytorch.org).  
```
# Example code for installing CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Step 5:
The following Python packages are necessary for running DeepMerge: argparse, h5py, numpy, pandas, pillow, tqdm, scipy, and scanpy. You can install them in the conda environment using the commands provided below:
```
pip install argparse==1.4.0
pip install h5py==3.1.0
pip install numpy==1.19.5
pip install pandas==1.3.4
pip install pillow==8.4.0
pip install tqdm==4.62.3
pip install scipy==1.7.1
pip install scanpy==1.8.2
```


## Preparing input for DeepMerge
DeepMerge’s main function takes raw count expression data (e.g., RNA, ADT, ATAC) in `.h5` format, as well as cell type labels and batch information in .csv format.

Here is an example of how to create a .h5 file from a count expression matrix data using the R environment. For your own dataset, you'll need to modify data and the specified save path accordingly.
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
write_h5(exprs_list = list(rna = data, h5file_list = "/DeepMerge/data/Rama/rna.h5")
```

### Example dataset

As an example, we provide the processed CITE-seq dataset by Ramaswamy et al. (GSM166489)[1] for a demonstration run, which is saved in ./DeepMerge/data/Rama/. Since the RNA data is large, users can download it from this [link](https://drive.google.com/drive/folders/1nYtSKP8b_ZBlykGwkY3hkjrTIHkc1v0B?usp=sharing). Users can either use the provided example dataset as input for DeepMerge or utilize their own datasets. Training on the demo dataset should take no more than 1 minute with a single GPU.

## Running DeepMerge with the example dataset
### Training the DeepMerge model (see Arguments section for more details).
You can either run the script file using the following command:
```
sh run.sh
```
or specify the parameters directly in the terminal:
```
python main.py --lr 0.02 --epochs 10 --batch_size 256 --hidden_modality1 185 --hidden_modality2 30  --modality1_path "./data/Rama/rna.h5"  --modality2_path "./data/Rama/adt.h5"  --cty_path "./data/Rama/cty.csv" --batch_path "./data/Rama/batch.csv" --dataset "Rama" --modality1 "rna" --modality2 "adt"
```

### Argument
Training dataset information
+ `--modality1_path`: path to the first modality containing all batches.
+ `--modality2_path`: path to the second modality containing all batches.
+ `--modality3_path`: path to the third modality containing all batches. (can be NULL if there is no data provided). 
+ `--cty_path`: path to the cell type labels of all data .
+ `--batch_path`: path to the batch information of all data.

Training and model config
+ `--batch_size`: Batch size.
+ `--epochs`: Number of epochs.
+ `--lr`: Learning rate.
+ `--z_dim`: Dimension of latent space.
+ `--hidden_modality1`: Hidden layer dimension of the first modality.
+ `--hidden_modality2`: Hidden layer dimension of the second modality.
+ `--hidden_modality3`: Hidden layer dimension of the third modality.

Other config
+ `--seed`: The random seed for training.
+ `--modality1`: The name of the first modality, e.g., RNA.
+ `--modality2`: The name of the second modality, e.g., ADT.
+ `--modality3`: The name of the third modality, e.g., ATAC.

## Visualisation
We utilize t-SNE visualization as an effective method to compare and evaluate the data both before and after being normalised by DeepMerge. This approach provides a clear visual representation of the data distribution and highlights the integration performance of DeepMerge. The t-SNE visualization of the original, unprocessed data is:

<img width=50% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/original.png"/>

From the results, it is clear that there are obvious batch effects present among different batches. To generate t-SNE visualizations of the data after normalization by DeepMerge, we can employ the ./qc/plot_TSNE.Rmd script by executing the following code snippets:
```
cd qc
Rscript plot_TSNE.Rmd
```
Then, the t-SNE plots illustrating the data after normalization by DeepMerge will be displayed, showcasing the effectiveness of DeepMerge in addressing batch effects. The following results show the normalized RNA, normalized ADT, and integrated embedding across modalities after DeepMerge.

<img width=58% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/batch_corrected.png"/>

To quantitatively evaluate the results, we can employ various metrics in both cell type conservation and batch mixing aspects, such as silhouette, NMI, and kBERT, which implemented in the [scib](https://scib.readthedocs.io/en/latest/) [2]. These metrics provide an objective assessment of the performance and effectiveness of the normalization and integration processes.

## Instructions on how to run DeepMerge on your data
To adapt these analyses for your dataset, modify the inputs to include your data, cell type labels, and batch information. The data should be in a `.h5` file format, containing all the data from one modality. If there are multiple modalities, you will need to have multiple data files with the same cell order. The cell type labels and batch files should be in `.csv` format and correspond to all the data present.

## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).

[2] Luecken MD, Büttner M, Chaichoompu K, Danese A, Interlandi M, Müller MF, Strobl DC, Zappia L, Dugas M, Colomé-Tatché M, Theis FJ. Benchmarking atlas-level data integration in single-cell genomics. Nature methods. 2022 Jan;19(1):41-50.

## License

This project is covered under the Apache 2.0 License.
