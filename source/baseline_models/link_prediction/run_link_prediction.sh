#!/bin/bash

cd ./OGB_GNN
python gnn.py --dataset ogbl-CD1-E_no1_spatial_edge_attr --batch_size 262144 --hidden_channels 32 --runs 1 --lr 0.025 --epochs 30
cd ..

cd ./OGB_MF
python mf.py --dataset ogbl-CD1-E_no1_spatial_edge_attr --batch_size 262144 --hidden_channels 32 --runs 1 --lr 0.025 --epochs 30
cd ..

cd ./OGB_MLP
python mlp.py --dataset ogbl-CD1-E_no1_spatial_edge_attr --batch_size 262144 --hidden_channels 32 --runs 1 --lr 0.025 --epochs 30
cd ..

# cd ./SEAL_OGB
# python seal_link_pred.py --dataset ogbl-CD1-E_no1_spatial_no_edge_attr --batch_size 262144 --hidden_channels 32 --runs 1 --lr 0.025 --epochs 30
# cd ..

# cd ./OGB_Node2Vec
# python ./OGB_Node2Vec/gnn.py --dataset ogbl-CD1-E_no1_spatial_no_edge_attr --batch_size 262144 --hidden_channels 32 --runs 1 --lr 0.025 --epochs 30
