#!/bin/awk -f
# roll_cols.awk
# inputs .tokdecs, outputs .tokmeasures

BEGIN {
    FS = "[ ]"; # define input field separator
    split(cols,out,":") #split string "cols" by ":" delim, saving output in array "out"
}
NR==1 { #for first line, or record number 1
    for (i=1; i<=NF; i++) #iterate over number of fields
        ix[$i] = i; #store field values of first line in "ix"
}

NR > 0 {
    first = 1;
    for (i in out)
        if (out[i] in ix)
            if (first) {
                printf "%s", $ix[out[i]];
                first = 0;
            }
            else {
                printf "%s%s", OFS, $ix[out[i]];
            }
    print "";
}
