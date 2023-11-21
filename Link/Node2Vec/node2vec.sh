#!/usr/bin/env bash

for data in "Cora_ML" "CiteSeer" 
do
  echo "Dataset:" $data
  for seed in 128 1024 4321 4399 42
  do
    python -W ignore Node2Vec/run.py --dataset $data --record --seed $seed --top 5
  done
done