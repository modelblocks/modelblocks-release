import sys, re

resources = {}

for path in sys.argv[1:]:
    with open (path, 'rb') as f:
        line = f.readline()
        while line:
            while line and not line.strip().startswith('define RESOURCE-DESCR'):
                line = f.readline()
            if line:
                descr = ''
                name = ''
                line = f.readline()
                while line and not line.strip().startswith('endef'):
                    if line.strip() != '':
                        descr += line.strip() + '  \n'
                        if line.startswith('NAME: '):
                            name = line.strip()[6:]
                    line = f.readline()
                resources[name] = descr
            line = f.readline()

print('''MODELBLOCKS RESOURCES
===========

This file provides descriptions and access details for each external resource not included in ModelBlocks. Modelblocks needs to know where to access external resources, so each such resource has an associated config/user-&ast;.txt file, which you will need to edit so that it contains the absolute path of that resource on your system.

''')
for r in sorted(resources.keys(), key=lambda x: x[4:] if x.startswith('The') else x):
    print(r)
    print('-'*50)
    print(resources[r])
    print('')
