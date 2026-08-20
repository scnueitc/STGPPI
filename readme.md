<!-- #region -->
# STGPPI

This study constructs residue graph features for protein–protein complexes and proposes a graph neural network framework-STGPPI, based on the TransformerConv operator and the SVD orthogonal initialization strategy, aiming to address the accuracy and generalization bottlenecks in protein–protein binding affinity prediction. 

## Download Data and Trained Weight

The generated datasets, consisting of 39-dimensional node features and 9-dimensional edge features, can be downloaded from the DATAs folder. 
The checkpoints folder contains the trained weights for both the STGPPI model and the PPB-Affinity baseline (trained on our dataset).

You can download dataset from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.22023965).

You can download PPB-Affinity Baseline model trained weight on our dataset from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.22024132).
## Train STGPPI
	```
	python train.py --data_path DATA/all.pt --save_dir ./STGPPI_FOLDER
	
	```
## Inference based on STGPPI
	```
	python your_script_name.py --data_path DATA/all.pt --checkpoint ./STGPPI.pth
	
	```

<!-- #endregion -->
