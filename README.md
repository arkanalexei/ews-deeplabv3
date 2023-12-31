# Description
Attempting to reproduce the results by Zenkl, R. as stated in this repository: https://github.com/RadekZenkl/EWS

All credits goes to Radek Zenkl, Radu Timofte, Norbert Kirchgessner, Lukas Roth, Andreas Hund, Luc Van Gool, Achim Walter and Helge Aasen

Original paper is provided for reference. You can access it at `fpls-12-774068.pdf`

## Setup the Data

Download the dataset from: https://www.research-collection.ethz.ch/handle/20.500.11850/512332

Create following folder structure:
```
EWS
└── data
     └── train
     └── validation
     └── test
     └── Data for Yu et al
          └── train
          └── validation
          └── test
     
```

# Installation (Windows)
1. Get and Install Anaconda/Miniconda
2. (Assuming Anaconda) Open Anaconda Navigator and open CMD.exe prompt
3. Run `conda env create -f env.yml`
4. Run `conda activate EWS`


# Quick GPU Test
If you are unsure if your setup can utilize a GPU, active the `EWS` environment and execute following snippet with python:

`import torch`

`print(torch.zeros(1).cuda())`

If no errors appear, you should be good to go. Otherwise, try reinstalling pytorch with different cuda version: https://pytorch.org/get-started/locally/

## Running 

In order to replicate the benchmarks, execute `main.py`. The resulting training, validation and testing metrics will be printed directly into the console. 