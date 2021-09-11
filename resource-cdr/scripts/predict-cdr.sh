get_base () 
{
basename $1 | rev | cut -d. -f2- | rev
}

get_suffix ()

{
echo "$1" | rev | cut -d. -f1 | rev
}

config_path="$1_outdir/config.ini"

pred_partition=$(get_base $2);
pred_partition=$(get_suffix $pred_partition);
pred_partition="${pred_partition/_part/}"
#pred_partition=$(echo $pred_partition | sed 's/fit/train/g' | sed 's/expl/dev/g' | sed 's/held/test/g' | sed 's/+/-/g')

cdr_dir=$(cat ../config/user-cdr-directory.txt)

export PYTHONPATH=$PYTHONPATH:$cdr_dir

python3 -m cdr.bin.predict $config_path -p "$2;$3"

cp "$1_outdir/CDR/preds_1.txt" "genmodel/$(get_base $1).$(echo $pred_partition)_prd.resmeasures"
cp "$1_outdir/CDR/obs_1.txt"  "genmodel/$(get_base $1).$(echo $pred_partition)_obs.resmeasures"
#cp "$3" "genmodel/$(get_base $1).$(echo $pred_partition)_obs.resmeasures"

#cat "$1_outdir/CDR/losses_mse_1.txt"
cat "$1_outdir/CDR/squared_error_1.txt"
# New version use squared_error_1.txt
#cat $1_outdir/CDR/loglik_*.txt

