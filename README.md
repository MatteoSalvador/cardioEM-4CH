# Whole-heart electromechanical simulations using Latent Neural Ordinary Differential Equations

This repository contains the code accompanying the paper [1]. We employ Latent Neural Ordinary Differential Equations (LNODEs) to learn the atrial and ventricular pressure-volume temporal dynamics of 400 four-chamber heart electromechanical simulations while spanning 43 physics-based model parameters that describe cell, tissue, whole-heart and cardiovascular system material properties and boundary conditions. We employ the surrogate model based on LNODEs to perform a global sensitivity analysis and robust parameter estimation with inverse uncertainty quantification for different test cases (`'TLV'`, `'Tatria'`, `'Tventricles'`, `'Tall'`).

## Installation

1. Install a conda environment containing all the required packages:

```bash
conda create -n envcardioEM-4CH python=3.8 numpy=1.21.5 matplotlib=3.5.1 pandas=1.3.4 scipy=1.7.3 mpi4py=3.1
conda activate envcardioEM-4CH
conda install tensorflow
conda install -c anaconda scikit-learn
pip install SALib
pip install --upgrade "jax[cpu]"
pip install numpyro
pip install diffrax
```

2. Clone this repository by typing:

```bash
git clone https://github.com/MatteoSalvador/cardioEM-4CH.git
```

3. Remember to activate the conda environment `envcardioEM-4CH` by typing `conda activate envcardioEM-4CH` (in case it is not already active from the installation procedure at point 1).

4. Unzip `dataset.zip`, which contains a tensor with all the 3D-0D numerical simulations (`simulations.pkl`), and a table with all the physics-based model parameters (`parameters.npy`).

5. **LNODEs training:** execute the Python script `train.py` with proper settings, such as the total number of Adam (`num_iters_ADAM`) and BFGS (`num_iters_BFGS`) epochs, `neurons`, `layers` and `states`.

6. **Global sensitivity analysis:** execute 4 different Python scripts sequentially (`SA_salib_1_params.py`, `SA_salib_2_execution.py`, `SA_salib_3_analysis.py`, `SA_salib_4_postprocessing.py`) with proper settings, such as the number of samples (`nsamples`), input (`inputfile`) and output (`outputfile`) files, required during the different stages.

7. **Parameter estimation with uncertainty quantification:** execute the Python script `run_PE_ANN.py` with a certain test case (`'TLV'`, `'Tatria'`, `'Tventricles'`, `'Tall'`) and proper settings, such as the folder containing the trained LNODEs (`ANN_folder`) and the number of trials (`n_trials`).

## Authors

- Matteo Salvador (<msalvad@stanford.edu>)
- Francesco Regazzoni (<francesco.regazzoni@polimi.it>)

## References

[1] M. Salvador, M. Strocchi, F. Regazzoni, C. Augustin, L. Dede', S. A. Niederer, A. Quarteroni. [Whole-heart electromechanical simulations using Latent Neural Ordinary Differential Equations](https://www.nature.com/articles/s41746-024-01084-x). *npj Digital Medicine* (2024).

[2] M. Salvador, F. Regazzoni, L. Dede', A. Quarteroni. [Fast and robust parameter estimation with uncertainty quantification for the cardiac function](https://www.sciencedirect.com/science/article/pii/S016926072300069X). *Computer Methods and Programs in Biomedicine* (2023).

[3] F. Regazzoni, M. Salvador, L. Dede', A. Quarteroni. [A Machine Learning method for real-time numerical simulations of cardiac electromechanics](https://www.sciencedirect.com/science/article/pii/S004578252200144X). *Computer Methods in Applied Mechanics and Engineering* (2022).
