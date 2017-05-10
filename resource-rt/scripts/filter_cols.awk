BEGIN {
    split(cols,out,":")
}
NR==1 {
    for (i=1; i<=NF; i++)
        ix[$i] = i;
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
