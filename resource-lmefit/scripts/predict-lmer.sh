#!/bin/bash

model="$1.rdata";
prdmeasures=$2;
resmeasures=$3;
../resource-lmefit/scripts/predict_lmer.r $model <(python3 ../resource-rt/scripts/merge_tables.py $prdmeasures $resmeasures evid)


