import sys, json, yaml

filebases = [
  'ParamVal',
  'KernelBlockDefs',
  'CompositeBlockDefs',
  'TargetBlockDefs'
]

for base in filebases:
  with open(base + '.yml', 'rb') as f:
    data = yaml.load(f)
  with open(base + '.js', 'wb') as f:
    f.write(base + ' = ')
    json.dump(data,
              f,
              sort_keys=True,
              indent=2,
              separators=(',', ': '))
