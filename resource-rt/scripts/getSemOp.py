import os, sys, re

ops = {
    "Aa": 0,
    "Ab": 0,
    "Ma": 0,
    "Mb": 0,
    "Ca": 0,
    "Cb": 0,
    "E": 0,
    "EnoV": 0,
    "EnoVarg": 0,
    "EnoVmod": 0,
    "EnoVg": 0,
    "EnoVh": 0,
    "V": 0,
    "Z": 0
}

def reset(operations):
    for op in operations:
        operations[op] = 0

print("word syncat basecat", end="")
for op in sorted(ops):
    print(" op" + op, end="")
print()

for line in sys.stdin:
    if line.startswith('----'): 
        reset(ops)
        emod = 0
        earg = 0
        # category of base of deepest derivation fragment
        basecat = line.split(':')[-1][:-2]
    if line.startswith('P'): syncat = line.split()[-1]
    if( line.startswith('F') ):
        if 'M' in line.split()[-1].split('&')[1]: emod = 1
        if re.search( '\d', line.split()[-1].split('&')[1] ): earg = 1
        if( 'V' in line.split()[-1].split('&')[1] ): ops['V'] = 1
        if( 'Z' in line.split()[-1].split('&')[1] ): ops['Z'] = 1
    if( line.startswith('W') ): w = line.split()[-1]
    if( line.startswith('J') ):
        if 'M' in line.split()[-1].split('&')[1]: emod = 1
        if re.search( '\d', line.split()[-1].split('&')[1] ): earg = 1
        if 'V' in line.split()[-1].split('&')[1]: ops['V'] = 1
        if 'Z' in line.split()[-1].split('&')[1]: ops['Z'] = 1
        if line.split()[-1].split('&')[2][0] in '0123456789': ops['Aa'] = 1
        if line.split()[-1].split('&')[3][0] in '0123456789': ops['Ab'] = 1
        if line.split()[-1].split('&')[2] == 'M': ops['Ma'] = 1
        if line.split()[-1].split('&')[3] == 'M': ops['Mb'] = 1
        if line.split()[-1].split('&')[2] == 'C': ops['Ca'] = 1
        if line.split()[-1].split('&')[3] == 'C': ops['Cb'] = 1
        if emod:
            ops['E'] = 1
            if not ops['V']:
                ops['EnoV'] = 1
                ops['EnoVmod'] = 1
        if earg:
            ops['E'] = 1
            if not ops['V']:
                ops['EnoV'] = 1
                ops['EnoVarg'] = 1
        if ops['EnoV']:
            if "-h" in syncat: ops['EnoVh'] = 1
            if "-g" in syncat or "-g" in basecat: ops['EnoVg'] = 1
        print(w + ' ' + syncat + ' ' + basecat, end="")  
        for op in sorted(ops):
            print(" {}".format(ops[op]), end="")
        print()
