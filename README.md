# Neural-networks-for-ERNIE-architecture
Neural networks for ERNIE architecture; trained with redesigned FC layers, quantized activations, noisy ensembles, etc. 

iPython file in the main directory includes results from our run, and we highly recommend you use IPython to understand how different blocks of code works. For this, Jupyter Notebook is needed. We used Anaconda which satisfies all the dependencies.

train_sample.py file is the python file version of the iPython notebook file "4layer784-256-256-256-10-splitted-3level-extLoad.ipynb"

More experiments with results from our run can be found in the iPython notebooks under /Other Experiments with Results in iPython Notebooks/ folder.

Experiment name convention is as follows:
4layer: 4-layer fully connected network
784-256-256-256-10: Size of hidden layers in the network.
-splitted-: this indicates that the network has its hidden layers is splitted to a few blocks (2-8), where each block is essentially a locally connected block and neurons in different blocks have no connection in between.

-3level- : number of leves used in quantization of activations
-extLoad-: Initial weights are externally loaded up to layer N-1 after training a N-1 layer network. After this initial parameters are loaded, the whole N-layer network is trained together. All pretrained models can be found in the folder /SavedModelVars . To run the code successfully, you need to move this folder under your /tmp/ directory.

More detailed comments are given inside iPython notebook "4layer784-256-256-256-10-splitted-3level-extLoad.ipynb" in the main directory.
