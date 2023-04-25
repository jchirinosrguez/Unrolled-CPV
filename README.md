# Deep-Unrolling
## Unrolled Deep Networks For Sparse Signal Restoration

*  **Author:** Mouna GHARBI
*  **Institution:** Centre de Vision Numérique, CentraleSupélec, Inria, Université Paris-Saclay
*  **Email:** mouna.gharbi@centralesupelec.fr

### File Organization

* Datasets:
  - Dataset 1: contains folders `Training`, `Validation` and `Test`, each of which has groundtruth and degraded Mass Spectrometry data with a constant noise level.
  
  - Dataset 2: contains folders `Training`, `Validation` and `Test`, each of which has groundtruth and degraded Mass Spectrometry data with a variable noise level.
    
* Unrolled-ISTA: contains python files: `ISTA_joint_architecture.py`, `ISTA_model.py`, `ISTA_network.py`, `ISTA_utils.py`, `module.py` and `runfile.py` to be run to train and test this method.
* Unrolled-Primal_Dual: contains python files `module.py`, `Network.py`, `PD_model.py` , `stepsize_architecture.py`, `utils.py` and `Runfile.py` to be run for training and testing.
* Unrolled-Half_Quadratic: contains python files `attached_architectures.py`, `mode.py`, `model.py`, `modules.py`, `network.py`, `tools.py` and `runfile.py` for training and testing the method.
    
