#!/bin/bash

python ./gnn/gnn.py --dataset ogbn-C57BL_6_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150 --use_sage
echo "Finished gnn.py"
sleep 30

python ./gnn/gnn.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150 --use_sage
echo "Finished gnn.py"
sleep 30
