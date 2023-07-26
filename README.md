# GPU_Reproducibility_Experiment
Experimental AutoEncoder setup for the quantification of random effects introduced by GPU utilisation 

## Conception
### Reproducibility Experiment
In order of quantifying and demonstrating the extent of random effects induced by atomic processes in a reproducible manner, a simple auto-encoder testing setup to be applied to freely available ECG data was created, with the testing setup’s idea being the utilisation of automated repetition of the same, deterministically set experiment, with each repetition initialising a new instance of a model class under identical conditions. 

### Bottleneck size Variation
To isolate the effect of layer size differences as much as possible, a three layer auto-encoder, consisting of just input-, bottleneck- and output-layer was created and only its bottleneck size was varied.

## Data
[The data](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) in question being a random selection of 5000 heartbeats of the same patient’s5 ECG measurements, taken from the BIDMC Congestive Heart Failure Database, originally published on PhysioNe and pre-processed as well as manually classified into one normal and four anomaly classes by Chen et al. (2014).

## Results
Results and generated models are uploaded here. Encoder outputs have not been uploaded, as they're reproducible with the model and data. 

## Requirements
Requirements are listed in a separate file. 
Tests were run on a Windows 11 x64 system (Version 10.0.22621, Build 22621) with a RTX 3070 laptop GPU.
GPU driver version 31.0.15.3640, CUDA Toolkit 11.8, cuDNN SDK 8.6.0 and Python 3.9.7. were used
