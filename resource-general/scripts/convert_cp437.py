#python3 convert_latin-1.py
#Convert stdin encoded as latin-1 to ascii

import io
import sys
import codecs

#sys.stdout = codecs.getwriter( 'utf8' )( sys.stdout )

input_stream = io.TextIOWrapper( sys.stdin.buffer, encoding='CP437' ) #'latin-1' )
#input_stream = sys.stdin
for line in input_stream:
  sys.stdout.write( line )  #.encode('utf-8').decode('latin-1') )
