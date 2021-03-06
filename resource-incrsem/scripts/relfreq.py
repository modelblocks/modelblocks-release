import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import model
import argparse
 
argparser = argparse.ArgumentParser(description='''
Counts and divides model-formatted files.
''')
argparser.add_argument('-n', '--no-normalization', dest='nonorm', action='store_true', help='Do not normalize')
args = argparser.parse_args()

counts = model.CondModel() 
for line in sys.stdin:
  predictor,_,response = line.strip('\n').rpartition(' : ')
  response,count = response.split(' = ') if ' = ' in response else (response,'1')
  counts[predictor][response] += float(count)

if not args.nonorm:
  for predictor in sorted(counts):
    denominator = sum([counts[predictor][val] for val in counts[predictor]])
    for response in sorted(counts[predictor]):
      print ( predictor + ' : ' + response + ' = ' + str(counts[predictor][response]/denominator) )
else:
  for predictor in sorted(counts):
    for response in sorted(counts[predictor]):
      print ( predictor + ' : ' + response + ' = ' + str(counts[predictor][response]) )


