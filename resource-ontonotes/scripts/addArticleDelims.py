import re
import sys

''' take stdin, sub any ==>.*<== lines with (U !ARTICLE).  if none, add (U !ARTICLE) at top '''

lines = sys.stdin.readlines()

for line in lines:
    if re.match(r'==>.*<==', line) is not None:
        delims_exist = True
        break
    delims_exist = False

if delims_exist == True:
    for line in lines: #sub (U !ARTICLE) for delim
        sys.stdout.write(re.sub(r'==>.*<==', "!ARTICLE", line)) #this outputs line if no match, right?
else:
    sys.stdout.write("!ARTICLE\n")
    for line in lines:
        sys.stdout.write(line)


