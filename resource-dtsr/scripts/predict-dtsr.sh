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
pred_partition=$(echo $pred_partition | sed 's/fit/train/g' | sed 's/expl/dev/g' | sed 's/held/text/g')

dtsr_dir=$(cat ../config/user-dtsr-directory.txt)

export PYTHONPATH=$PYTHONPATH:dtsr_dir

python3 -m dtsr.bin.predict $config_path -p $pred_partition

cat "$1_outdir/DTSR/losses_mse_$pred_partition.txt"

