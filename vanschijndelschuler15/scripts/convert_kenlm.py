import sys

with open(sys.argv[1],'r') as f:
    lines = f.readlines()

line = lines[0].split('\t')
sys.stdout.write(str(line)+'\n')
