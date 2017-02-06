# break line at terminal punctuation mark (but not if next word is lower case)...
##s/([\.\!\?][\.\"\'\)]*) +(?![ a-z])/\1\n/g;
s/([\.\!\?][\.\"\'\)]*) +(?=[\`\'\(\[]*[A-Z0-9])/\1\n/g;

# translate initial and final quotes...
s/(?:\"|\'\')(?![ \t\n\.,!?])(?!\))/\`\`/g;
s/([^ ])\"/\1''/g;
s/(?<![\w\,\.\!\?\'])\'(?! |\')/\` /g;

# separate initial, medial, and final punctuation marks...
s/(?<!\w)(\(|\[|\`+|\$)(?! )/\1 /g;
s/(?<! )(\-+|\$|\(|\)|\`\`|\'\'|\[|\])(?! )/ \1 /g;
s/(\-+|,|;|:|%|\)|\]|\'+|\.+|\!|\?)(?![A-Za-z0-9])/ \1/g;

# separate possessives and contractions...
s/(\'s|\'d|\'re|\'ll|n\'t) / \1 /g;

# if next word is lower case, undo spacing...
s/ +([\.\!\?][\.\"\'\)]*) +(?=[\`\'\(\[ ]*[a-z])/\1 /g;

# lowercase...
#y/[A-Z]/[a-z]/;

# translate special characters...
s/\(/-LRB-/g;
s/\)/-RRB-/g;
s/\[/-LRB-/g;
s/\]/-RRB-/g;
s/\@/\!at\!/g;

# remove extra spaces in beginning / middle / end...
s/^ +//g;
s/ *\n */\n/g;
s/  */ /g;
s/ *$//g;
