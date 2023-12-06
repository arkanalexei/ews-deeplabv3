# Description
Attempting to reproduce the results by Zenkl, R. as stated in this repository: https://github.com/RadekZenkl/EWS

All credits goes to Radek Zenkl, Radu Timofte, Norbert Kirchgessner, Lukas Roth, Andreas Hund, Luc Van Gool, Achim Walter and Helge Aasen3


# Installation
1. Get Anaconda/Miniconda (whichever you prefer) https://www.anaconda.com/download
2. Run `conda env create -f env.yml`
3. `conda activate EWS`


# Quick GPU Test
If you are unsure if your setup can utilize a GPU, active the `EWS` environment and execute following snippet with python:

`import torch``

`print(torch.zeros(1).cuda())``

If no errors appear, you should be good to go. Otherwise, try reinstalling pytorch with different cuda version: https://pytorch.org/get-started/locally/