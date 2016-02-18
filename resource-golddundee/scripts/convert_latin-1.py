#python3 convert_latin-1.py
#Convert stdin encoded as latin-1 to ascii

import io
import sys

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')

for line in input_stream:
  sys.stdout.write(line)
