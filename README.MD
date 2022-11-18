# Discovering Locally Maximal Bipartite Subgraphs

This repository contains the code and data for the paper "Force-Directed Layout of Order Diagrams using Dimensional Reduction".

## Instructions
- Create the folder "data" with the datasets.
- Install requirements from requirements.txt.
- Install the MiniSat binary from [here](http://minisat.se/MiniSat.html).
- Run bipartite.py to run all experiments. Results are output to the data.db.

## File Structure
| Filename                  | Description                                |
| ------------------------- | ------------------------------------------ |
| data/                     | Place the downloaded datasets here.        |
| bipartite.py              | Runs the experiments.                      |
| data.db                   | Results of the experiments are saved here. |
| generate_random_graphs.py | Generates the Random dataset.              |
| requirements.txt          | Used to install requirements.              |

## Datasets
| Dataset name | How to get it                                                                 |
| ------------ | ----------------------------------------------------------------------------- |
| DAGmar       | Download from [here](http://www.graphdrawing.org/download/DAGmar.tgz).        |
| North        | Download from [here](http://www.graphdrawing.org/download/north-graphml.tgz). |
| Random       | Generate with the provides script "generate_random_graphs.py".                |
| Rome         | Download from [here](http://www.graphdrawing.org/download/rome-graphml.tgz).  |