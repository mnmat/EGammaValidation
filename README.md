# EGammaValidation

This framework contains tools to analyse the input features and the performance of the EGRegresTrainerLegacy.

Setup environment (for SWAN 105a):

```
. /cvmfs/sft.cern.ch/lcg/views/LCG_105a_swan/x86_64-el9-gcc13-opt/setup.sh
```

# Analysis scripts

To analyse input features and the results of the BDT:

```
python3 analyse_features.py
python3 produce_plots.py 
```

The `analyse_features.py` script produces histograms of the input feature. It also produces correlation plots between intput and target.

The `produce_plots.py` script contains tools to create histograms of the different energies, regressed, raw, gen. It also contains tools to perform different fits such as Cruijff and DSCB. The parameters of these distributions can also be plotted. It will produce plots for different energy and eta bins.


# Compare different samples

To compare samples, use:

```
python3 compare_features.py
python3 compare_samples.py
```

