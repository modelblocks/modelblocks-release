#!/bin/bash

model="/fs/project/schuler.77/oh.531/$1.rdata";
prdmeasures=$2;
resmeasures=$3;
../resource-gamfit/scripts/predict_gam.r $model <(python3 ../resource-rt/scripts/merge_tables.py $prdmeasures $resmeasures evid)


