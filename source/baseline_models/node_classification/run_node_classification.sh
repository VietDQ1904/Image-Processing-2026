#!/bin/bash

echo "Start running scripts..."

python ./gnn/gnn.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 64 --runs 1 --epochs 150
echo "Finished gnn.py"
sleep 30

python ./cluster_gcn/cluster_gcn.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150
echo "Finished cluster_gcn.py"
sleep 30

python ./mlp/mlp.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 64 --runs 1 --epochs 150
echo "Finished mlp.py"
sleep 30

python ./node2vec/node2vec.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025
echo "Finished node2vec.py"
sleep 30

python ./graph_saint/graph_saint.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150
echo "Finished graph_saint.py"
sleep 30

python ./sign/sign.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150
echo "Finished sign.py"
sleep 30

python ./spec_mlp_w_cs/mlp_cs.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 64 --runs 1 --epochs 150
echo "Finished spec_mlp_w_cs.py"
sleep 30

echo "All scripts done!"