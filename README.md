# GMAN

## Credits
original paper: https://arxiv.org/abs/1911.08415 

## About
The original codebase of GMAN was designed for PEMS and METR LA. This modified version is edited to run of 28 sensors data of the Las Vegas I15 dataset. 

The code requires the data for 28 sensors in HDF format. For the STAtt block the code requires a graph embeddings. For this purpose we made a linear graph for 28 sensors being the edge and each vertex having equal weight of 1. The graph is represented as adjacency matrix, Adj(LAS-28). 

We used the generateSE script which uses node2vec algorthrim to get the spatial embedding file, SE(LAS-28).

##  Requirements:
* Python 3.6
* PyTorch
* Pandas
* Matplotlib
* Numpy
* Wheel

## Parameters to be investigated:
* no of STAtt blocks
* no history steps
* no prediction steps
* no attention heads
* dims of each head attention outputs
---
* shape of the graph
* weights of the edges
* dynamic graph
* addition of external factors such as neighbouring traffic, weather
---
* node2vec:
  * p/q
  * num_walks 
  * walk_length 
  * dimensions 
  * window_size 

