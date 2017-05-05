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
      elif line.strip().startswith('def KernelBlock'):
        line = M.readline()
        while line and not line.startswith('endef'):
          KernelBlockDefs += line
          line = M.readline()
      elif line.strip().startswith('def CompositeBlock'):
        line = M.readline()
        while line and not line.startswith('endef'):
          CompositeBlockDefs += line
          line = M.readline()
      elif line.strip().startswith('def TargetBlock'):
        line = M.readline()
        while line and not line.startswith('endef'):
          TargetBlockDefs += line
          line = M.readline()
      else:
        line = M.readline()

with open('ParamVal.yml', 'wb') as f:
  f.write(ParamVal)
with open('KernelBlockDefs.yml', 'wb') as f:
  f.write(KernelBlockDefs)
with open('CompositeBlockDefs.yml', 'wb') as f:
  f.write(CompositeBlockDefs)
with open('TargetBlockDefs.yml', 'wb') as f:
  f.write(TargetBlockDefs)

