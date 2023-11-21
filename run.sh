#!/usr/bin/env bash
for model in "GCN" "GAT" "GraphSAGE" "APPNP" "GIN" 
do
  for data in "Cora" "Citeseer" "Pubmed" "LastFM" "APh"
  do
    for seed in 1234 4321 42 4399 128
    do
      python -W ignore baseline/m3s.py --multiview --dataset $data --model $model --iter 20 --seed $seed
      python -W ignore baseline/drgst.py --dataset $data --model $model --seed $seed
    done
  done
done

for model in "GCN" "GAT" "GraphSAGE" "APPNP" "GIN"
do
  for seed in 1234 4321 42 4399 128
  do
    python -W ignore main_node.py --multiview --dataset Cora --model $model --top 100 --iter 25 --aug_drop 0.1 --seed $seed
    python -W ignore main_node.py --multiview --dataset Citeseer --model $model --top 70 --iter 30  --aug_drop 0.05 --seed $seed
    python -W ignore main_node.py --multiview --dataset Pubmed --model $model --top 1000 --iter 20 --aug_drop 0.1 --seed $seed
    python -W ignore main_node.py --multiview --dataset LastFM --model $model --top 400 --iter 20 --aug_drop 0.2 --seed $seed
    python -W ignore main_node.py --multiview --dataset APh --model $model --top 200 --iter 35 --aug_drop 0.1 --seed $seed
  done
done