#!/bin/bash

m=$(python ../resource-lmefit/scripts/infer_lrt_modelnames.py $*);
echo $m
../resource-lmefit/scripts/lmefit2lrtsignif.r $m;

