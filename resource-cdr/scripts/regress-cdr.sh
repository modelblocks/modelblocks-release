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

base_filename="$dname$(basename $1 _part.prdmeasures)_$(basename $3 .cdrform)$predsSTR"
outdir="${base_filename/_part\./\.}"
outfile="${base_filename/_part\./\.}"
outdir+="_cdr.fitmodel_outdir";
outfile+="_cdr.fitmodel";

cat $3 | python3 ../resource-cdr/scripts/baseform_to_config.py $1 $2 $outdir $preds

cdr_dir=$(cat ../config/user-cdr-directory.txt)

export PYTHONPATH=$PYTHONPATH:$cdr_dir

python3 -m cdr.bin.train $outdir/config.ini
cp $outdir/CDR/summary.txt $outfile

