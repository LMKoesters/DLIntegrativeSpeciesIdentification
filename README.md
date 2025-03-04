# Data Fusion for Integrative Species Identification Using Deep Learning
## Contents
- [Introduction](#introduction)
- [Data](#data)
- [Setup environment](#setup-environment)
- [Example usage](#example-usage)
- [Detailed description of scripts](#detailed-description-of-scripts)


## Introduction
This repository contains the code needed to reproduce the results presented in the paper [Data Fusion for Integrative Species Identification Using Deep Learning](https://doi.org/10.1101/2025.01.22.634270). 
Within our paper, we systematically showed and explained, for the first time, that optimizing
the preprocessing and integration of molecular and image data offers significant benefits,
particularly for genetically similar and morphologically indistinguishable species,
enhancing species identification by reducing modality-specific failure rates and information
gaps. Our results can inform integration efforts for various organism groups, improving
automated identification across a wide range of eukaryotic species.

## Data
All data required to reproduce our results can be found on Dryad (link to be added soon).

## Setup environment
Python environment setup:

```bash
# 1st option
# create environment with all dependencies via conda
conda env create -f environment.yml

# 2nd option
# create and activate virtual python environment
conda create -n integrative_dl
conda activate integrative_dl
# install required packages
python3 -m pip install -r requirements.txt
```

For BLAST install BLAST+ v2.15.0. You can either install it into your conda environment (https://anaconda.org/bioconda/blast) or by following the instructions at https://www.ncbi.nlm.nih.gov/books/NBK569861/.

In addition, you will need to have R installed (v4.3.1). The pipeline will install the required packages itself. 

## Example usage

Let us assume we want to train our models on a dataset that contains data of the _Asteraceae_ family. We will crawl DNA data and images from GenBank, BOLD, and GBIF first by using the scripts within _src/dataset_collection_.
```bash
cd src/01_dataset_collection
python main.py --query 'Asteraceae' --project-dir ${YOUR_PROJECT_DIRECTORY}
```
> [!WARNING]  
> DO NOT RUN qc.add_train() the first time you are prompted to manually check the images.

We will apply manual image filtering (the script will prompt us to do so) by starting a jupyter notebook server and running the code within the second to last cell. 
Don't forget to add your job name, the marker that the pipeline chose during dataset collection and the directory that contains your project in the respective fields before running the cell.

```bash
jupyter notebook quality_filtering.ipynb
```
Afterwards, we will run the main script within _src/01_dataset_collection_ again. Keep in mind that manual filtering needs to be done twice and that, both times, the main scripts needs to be run afterwards.

Next, we will train models based on a traditional train/val split (where two samples per species will be used for validation) and LOOCV.
```bash
cd src/02_classification
python main.py --job-id 'Asteraceae' --root-dir ${YOUR_PROJECT_DIRECTORY} 
```

The results will be stored in a results_${run_index}.tsv file. The run index is determined by an argument that can be given to _src/02_classification/main.py_ to run multiple LOOCV runs in parallel on the same machine - with shared logs to allocate samples to each of the processes. 
To leverage this functionality, _--runs_ needs to be set to >1, i.e. the number of parallel LOOCV runs. The argument _--run-index_ then determines the index of the current run. For instance, `--runs 4 --run-index 2` means that there are 4 parallel runs in total and the current run is the third (due to 0-based indexing).

To evaluate the influence of, e.g., DNA sequence length within the training dataset, we can leverage `src/03_GLM/ModelOptimizer.R`. For a more flexible approach to regression modeling, consider using [LazyModeler](https://github.com/LMKoesters/LazyModeler), which builds upon _ModelOptimizer.R_.

To generate plots and statistics based on the results, we can run each of the R scripts within _src/04_results_evaluation_. Again, the required packages will be automatically installed. Remember to set the directory that the results are stored in (the variable is called _base_dir_).   
 

## Detailed description of scripts
### Dataset collection
The scripts in this folder are responsible for collecting a dataset based on a query (e.g., a family). Records will be crawled from BOLD and GENBANK, while images are gathered from BOLD, GBIF, and (in case of _in situ_ images) a local Flora Capture folder. The only scripts that need to be actively started are the main script and the jupyter notebook for manual quality filtering. The main script will ask for the user to check the quality at some point. The jupyter notebook will then need to be used and the main script restarted. There are in total 2 times that the jupyter notebook will be needed.

#### R/genetic_distance.R
Calculates a distance matrix based on sequence identity for a given alignment file and writes the matrix in tsv format to a provided output path.

#### quality_filtering.ipynb
Jupyter notebook used to semi-manually filter images by iteratively printing images and asking for (dis)approval. This is the only script apart from main.py that needs to be actively run. When run for the second time, the script will add a column used for splitting training and validation data.

#### barcode_filtering.py
Contains methods for filtering DNA data. This includes filtering based on clusters and their gap and SNP content, which uses vsearch, snp-sites and vcftools as third-party tools. This script also provides the information needed to choose the best barcode based on the metrics described in the Methods section.

#### commons.py
Includes methods that are shared between multiple classes. Besides a log method, methods in this script are mainly related to image crawling, downloading, and checking.

#### data_bold.py
This script is responsible for downloading data from BOLD and formatting the data for easier access by methods further down the pipeline.

#### data_ncbi.py
Similarly, this script is responsible for downloading data from NCBI and formatting the data for easier access by methods further down the pipeline.

#### data_padding.py
We search BOLD for combined records, i.e., records with a genetic sequence and an attached image. NCBI, however, does not provide images. Therefore, we need to add images to the records that are not linked to an image already. In this study, we used Flora images alongside in situ images for the Poaceae dataset. Thus, there are methods for searching for images on disk. Other methods crawl images from GBIF, either by searching for in situ iNaturalist images or by searching for preserved specimens and attaching the URLs to the records. When BOLD records are removed based on their barcode during barcode filtering, the attached images are re-used where possible.

#### dataset.py
This file contains all methods handling the compilation of a dataset. It starts both BOLD and NCBI data crawling, applies the threshold multiple times, chooses the best marker based on the results of the barcode filtering, and checks for image and GenBank accession duplicates.

#### main.py
Main script file and the only script apart from quality_filtering.ipynb that needs to be actively run by the user. This script takes all user arguments for running the complete pipeline to create a dataset based on a) a given taxonomic group, b) a file with GenBank accession numbers, or c) a BOLD container. For detailed information on the available user arguments, please refer to the file or run main.py --help.

#### server_prep.py
The methods within this script prepare the genetic and image data for consumption by the methods in 02_classification. This preparation includes the encoding and aligning/SNP-reduction of the genetic sequences.

#### stats.py
Responsible for printing information about a given dataset. This includes genetic distances within the dataset and overall dataset information such as the number of species.

### Classification
This folder contains the scripts for the training of the ML models for species identification. Unless random forest is not to be applied, the random forest grid search needs to be run first. Then, the main script is the one that needs to be actively started. All other scripts are then automatically included and run.

#### barcodejpg_dataset.py
Contains the dataset class for the image+DNA dataset that is given to the DL model. If the DL is trained on either the DNA or both data types, the DNA is directly loaded into memory to save time during training.

#### blast.py
Contains methods to run BLAST both in with a traditional train-val split and as leave-one-out cross validation.

#### early_stopping.py
Small class that adds early stopping to training the DL model.

#### main.py
Main script file and the only script that needs to be actively run by the user.

#### model_bar_resnet_sequential_embedding.py
Same as model_bar_resnet.py with small adjustments to handle ordinal/sequential DNA encoding. The main adjustment is the addition of a layer that transforms the fractional encoding of the DNA into a DL-driven ordinal encoding.

#### model_bar_resnet.py
Here, the DL model that handles the DNA is defined. The content mainly originates from the pytorch GitHub, with small adjustments to the channel size and the classifier.

#### model_barimg.py
This script defines the model used for the fusion experiments. It uses the CNNs of both the barcode/DNA and image models, defines the classifier and how the two input types are forwarded through the network.

#### model_img.py
This script defines the model used for the images. It directly uses the pytorch-defined ResNet-50 and adds a custom classifier.

#### random_forest_grid_search.py
This script is not automatically run when calling main.py. It is used to determine the parameters to be used during the main training by applying grid search and training and evaluating on the training subsets of the datasets.

#### train_parent.py
This script contains a wrapper class that starts both traditional and leave-one-out training. For LOO, it automatically prepares the dataset while considering the subset size of 4 that is used in this study. It also checks which of the preprocessings performed best and skips already trained-on validation indices. 

#### train.py
This is the main script for training. It first checks if the specific training was already done or is currently handled by another process. Then it sets up the models and model parameters, trains the specified models (separate and fused), saves the models and results, and creates a history plot visualizing information about the accuracy and loss during each epoch.


### GLM - ModelOptimier.R
Here, the R script for automatic GLM model simplification resides. It can remove autocorrelations and uses a list of coefficients sorted by their relevance in order to remove not significantly correlated predictors.


### Results evaluation
Contains the scripts that were used for plot creation.

#### barcode_preprocessing.R
Comparison of DNA preprocessing methods (arrangement & encoding).

#### confusion_w_genetic_distances.R
Analysis of genetic distances of samples that were misidentified by at least one model (either unimodal or multimodal). Contains information on 

#### fusion_methods.R
Comparison of unimodal models and multimodal models using different fusion methods.

#### supporting_confusion_heatmap.R
Confusion rates and bias towards either inter- or intrageneric confusion alongside information on sample sizes.

