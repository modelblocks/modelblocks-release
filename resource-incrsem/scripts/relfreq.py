import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import model

counts = model.CondModel() 
for line in sys.stdin:
  predictor,response = line.strip('\n').split(' : ')
  counts[predictor][response] += 1

for predictor in sorted(counts):
  denominator = sum([counts[predictor][val] for val in counts[predictor]])
  for response in sorted(counts[predictor]):
    print ( predictor + ' : ' + response + ' = ' + str(counts[predictor][response]/denominator) )


