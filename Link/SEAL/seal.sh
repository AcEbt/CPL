#!/usr/bin/env bash

do
  for seed in 128 1024 4321 4399 42
  do
    python -W ignore SEAL/run.py --dataset Actor --record --use_feature --seed $seed
  done
done