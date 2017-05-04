import sys, re, os

Makefiles = []
ParamVal = ''
KernelBlockDefs = ''
CompositeBlockDefs = ''
TargetBlockDefs = ''

for root, dirs, files in os.walk('../'):
  for name in files:
    if name.endswith('Makefile'):
      Makefiles.append(os.path.join(root,name))

for path in Makefiles:
  with open(path, 'rb') as M:
    line = M.readline()
    while line:
      if line.strip().startswith('def ParamVal'):
        line = M.readline()
        while line and not line.startswith('endef'):
          ParamVal += line
          line = M.readline()
      elif line.strip().startswith('def KernelBlockDefs'):
        line = M.readline()
        while line and not line.startswith('endef'):
          KernelBlockDefs += line
          line = M.readline()
      elif line.strip().startswith('def CompositeBlockDefs'):
        line = M.readline()
        while line and not line.startswith('endef'):
          CompositeBlockDefs += line
          line = M.readline()
      elif line.strip().startswith('def TargetBlockDefs'):
        line = M.readline()
        while line and not line.startswith('endef'):
          TargetBlockDefs += line
          line = M.readline()
      else:
        line = M.readline()

#print('ParamVal')
with open('ParamVal.yml', 'wb') as f:
  f.write(ParamVal)
#print('')
#print('KernelBlockDefs')
#print(KernelBlockDefs)
#print('')
#print('CompositeBlockDefs')
#print(CompositeBlockDefs)
#print('')
#print('TargetBlockDefs')
#print(TargetBlockDefs)
