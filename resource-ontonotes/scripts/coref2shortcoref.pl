
# remove empty heads
s/\*[^ \n<>]*\*?(-[0-9]+)?//g;
s/ 0 / /g;

# tokenize hyphens
s/(\w)-(\w)/\1 - \2/g;
# un-tokenize hyphens that are part of '-AMP-'
s/ - AMP - /-AMP-/g;

# move coref id to end and turn recursive corefs anglebrackets into braces
while ( s/<COREF ID="([0-9]+)"[^>]+> *([^<]*?) *<\/COREF>/\{COREF\}\2\{\/COREF\1\}/g ) { }

# turn corefs back into anglebrackets
s/{/</g;
s/}/>/g;

