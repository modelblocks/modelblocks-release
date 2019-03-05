#!/bin/bash

prdmeasures=$1;
resmeasures=$2;
bform=$3;
preds=${*:4:$#};

if [ -z "$preds" ]; then
	preds_STR="";
	preds_add_STR="";
	preds_ablate_STR="";
else
	until [ -z "$4" ]
	do
		if [ -z "$preds_STR" ]; then
			preds_STR=($4);
		else
			preds_STR+=($4);
		fi;
		if [[ $4 =~ ~.* ]]; then
			new=${4:1};
			if [ -z "$preds_ablate" ]; then
				preds_ablate=($new);
			else
				preds_ablate+=($new);
			fi;
		else
			new=$4;
		fi;
		if [ -z "$preds_add" ]; then
			preds_add=($new);
		else
			preds_add+=($new);
		fi;
		shift
	done
	preds_STR=$(IFS="_" ; echo "${preds_STR[*]}");
        echo "$preds_STR" > /dev/stderr
	preds_add_STR=$(IFS=\+ ; echo "${preds_add[*]}");
	preds_add_STR="-A $preds_add_STR";
        echo "$preds_add_STR" > /dev/stderr
	if [ ! -z "$preds_ablate" ]; then
		preds_ablate_STR=$(IFS=\+ ; echo "${preds_ablate[*]}");
		preds_ablate_STR="-a $preds_ablate_STR";
	else
		preds_ablate_STR="";
	fi;
fi;

dname_src=$(dirname $prdmeasures)
if [ -n "$dname_src" ]
then
	dname="$dname_src/";
else
	dname="";
fi

base_filename="$dname$(basename $prdmeasures _part.prdmeasures)_$(basename $bform .lmerform)_$preds_STR"
outdir="${base_filename/_part\./\.}"
outfile="${base_filename/_part\./\.}"
outfile+="_lmer.fitmodel.rdata";
corpusname=$(cut -d'.' -f1 <<< "$(basename $prdmeasures)");

../resource-lmefit/scripts/evmeasures2lmefit.r <(python ../resource-rt/scripts/merge_tables.py $prdmeasures $resmeasures subject docid sentid sentpos word) $outfile -b $bform $preds_add_STR $preds_ablate_STR -c $corpusname -e


