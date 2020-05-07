#!/bin/bash

preds=${*:4:$#};

predsSTR=$(printf "_%s" ${preds[@]});
dname_src=$(dirname $1)
if [ -n "$dname_src" ]
then
	dname="$dname_src/";
else
	dname="";
fi

base_filename="$dname$(basename $1 _part.prdmeasures)_$(basename $3 .dtsrform)$predsSTR"
outdir="${base_filename/_part\./\.}"
outfile="${base_filename/_part\./\.}"
outdir+="_dtsr.fitmodel_outdir";
outfile+="_dtsr.fitmodel";

cat $3 | python3 ../resource-dtsr/scripts/baseform_to_config.py $1 $2 $outdir $preds

dtsr_dir=$(cat ../config/user-dtsr-directory.txt)

echo $dtsr_dir

export PYTHONPATH=$PYTHONPATH:$dtsr_dir

python3 -m dtsr.bin.train $outdir/config.ini
cp $outdir/DTSR/summary.txt $outfile

