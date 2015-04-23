#python compileComplexity.py toksfile parsertoksfile
# Compiles a corpus of parser-tokenized complexity metrics to some coarser tokenization scheme
#toksfile: a file with one word per line
#parsertoksfile: a file with one parser-tokenized word per line

import re
import sys

if len(sys.argv) != 3:
  raise InputError(sys.argv,"Insufficient arguments")

DEBUG = False
restart = re.compile('^WORD')

pcorpus = [] #parser-tokenized corpus

with open(sys.argv[2],'r') as pfile:
  #load parser tokenized corpus
  for line in pfile.readlines():
    sline = line.strip().split()
    if sline != []:
      pcorpus.append(sline)

output = [pcorpus[0]+['parsed']] #begin by storing the complexity header in the output corpus
rowlen = len(output[0])-1 #the length a row of complexity metrics should be
pix = 1 #position within the parser-tokenized corpus

with open(sys.argv[1],'r',encoding="latin-1") as gfile:
  for line in gfile.readlines():
    #for each line in the gold file we're attempting to match...
    if restart.match(line) or line.strip() == '':
      #if there's no word for this line, skip it
      continue

    PARSED = True #was this row part of a successful parse?

    row = pcorpus[pix] #create a new row to begin appending to
    if rowlen > len(row) or '-nan' in row or 'nan' in row:
      PARSED = False #parse failure
      row += [0]*(rowlen - len(row)) #fill in values for the misparsed row #do we want this to be 0? Do we care what it is?
    pix += 1 # This looks like it's in the wrong place, but it's not; pix marks the NEXT parsed index to check, so it increments here since we just used pix
    target = line.split()[0]
    while row[0] != target:
      #while the parser-tokenized word conglomerate isn't the word we're looking for...
      # add the next row of parser-tokenized info to the conglomerate
      if DEBUG:
        sys.stderr.write('Current: '+str(row[0])+'\n Target:'+str(target)+'\n')
      for i in range(rowlen - 1):
        # don't sum the final metric because that's sentid
        if rowlen > len(pcorpus[pix]) or '-nan' in pcorpus[pix] or 'nan' in pcorpus[pix]:
          #parse failure
          PARSED = False
          row[0] += pcorpus[pix][0]
          break
        else:
          if i == 0:
            #treat as strings
            row[i] = row[i] + pcorpus[pix][i]
          else:
            #treat as floats
            row[i] = str(float(row[i]) + float(pcorpus[pix][i]))
      pix += 1
    #now that the parser-tokenized conglomerate matches the gold word we wanted...
    output.append(row + [str(PARSED)])

#output the compiled corpus
for row in output:
  sys.stdout.write(' '.join(row)+'\n')
