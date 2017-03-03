#python spargeComplexity.py [-o] [-id id_file]
#extract complexity from stdin (mixed complex/output)
#-o: extract output from stdin (instead of complexity)
#-id: output sentence ids from the given id_file
import re
import sys

REPORT_COMPLEX = True
idlist = []

if len(sys.argv) > 1:
  for i,arg in enumerate(sys.argv):
    if arg == '-o':
      REPORT_COMPLEX = False
    if arg == '-id':
      try:
        with open(sys.argv[i+1], 'r') as idfile:
          for l in idfile.readlines():
            idlist.append(l.strip())
      except:
        # we'll have to generate sentids on the fly
        pass

complex = [] #the complexity corpus
output = [] #the output corpus

retoggle = re.compile('^------') #the trigger to switch between complexity and output
NOWOUTPUT = False #a flag to denote whether current input is complexity or output
IN_COMPLEX = True #a flag to denote whether to output the next complexity line or not

idctr = 0

line = sys.stdin.readline()
# Check if this is actually mixed EFABP output
if line.strip().startswith('word'):
  complex.append(line.strip() + ' sentid\n')
  for line in sys.stdin.readlines():
    if line.strip() == "":
      #remove blank lines
      continue
    if retoggle.match(line):
      #a new input format, so record the change and move on
      idctr += 1
      if NOWOUTPUT:
        output.append(line) #save the time/length/id info in the output
        NOWOUTPUT = False
        IN_COMPLEX = False #don't output the complexity header if it's not the first one seen
      else:
        complex.pop() #remove the last element since it's the "first word of the next sentence"
        NOWOUTPUT = True
    else:
      if NOWOUTPUT:
        #add output lines to the output corpus
        output.append(line)
      else:
        #add complexity lines to the complexity corpus
        if IN_COMPLEX:
          #only allow the first observed complexity header along with the complexity metrics
          if idlist == []:
            complex.append(line.strip() + ' ' +str(int(idctr/2)) + '\n')
          else:
            complex.append(line.strip() + ' ' + idlist[int(idctr/2)] + '\n')
        else:
          #subsequent complexity headers have been successfully skipped
          IN_COMPLEX = True
# If not mixed EFABP output, just pass through
else:
  output.append(line)
  for line in sys.stdin.readlines():
    output.append(line)

if REPORT_COMPLEX:
  #output the complexity corpus
  for l in complex:
    sys.stdout.write(l)
else:
  #output the output corpus
  for l in output:
    sys.stdout.write(l)
