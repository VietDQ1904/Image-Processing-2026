#!/bin/bash

python ./cluster_gcn/cluster_gcn.py --dataset ogbn-BALBc_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150 --batch_size 2 --num_partitions 40
echo "Finished.."
sleep 30

python ./cluster_gcn/cluster_gcn.py --dataset ogbn-CD1_E_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150 --batch_size 2 --num_partitions 40
echo "Finished.."
sleep 30

python ./cluster_gcn/cluster_gcn.py --dataset ogbn-C57BL_6_no1_pb_minRadiusAvg --lr 0.025 --hidden_channels 32 --runs 1 --epochs 150 --batch_size 2 --num_partitions 40
echo "Finished.."
sleep 30

