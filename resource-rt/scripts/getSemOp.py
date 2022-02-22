import os, sys, re

ops = {
    "Aa": 0,
    "Ab": 0,
    "Ma": 0,
    "Mb": 0,
    "Ca": 0,
    "Cb": 0,
    "E": 0,
    "V": 0,
    "Z": 0,
    "EnoVnoHO": 0,
    "EnoVhNoHO": 0,
    "EnoVHO": 0
}

for i in {"", "g", "h"}:
    for j in {"", "arg", "mod"}:
        for k in {"", "lex", "grm"}:
            ops["EnoV"+i+j+k] = 0


def reset(operations):
    for op in operations:
        operations[op] = 0


print("word syncat basecat", end="")
for op in sorted(ops):
    print(" op" + op, end="")
print()

for line in sys.stdin:
    if line.startswith("----"): 
        reset(ops)
        emod = 0
        earg = 0
        elex = 0
        egrm = 0
        g = 0
        h = 0
        hO = 0
        # category of base of deepest derivation fragment
        basecat = line.strip().split(":")[-1][:-1]
        # stack ending in :: means deepest base cat is ":"
        if not basecat:
            basecat = "COLON"
    if line.startswith("P"): 
        syncat = line.split()[-1]
        # to avoid pesky downstream errors reading quotation marks in CSV
        if syncat == '"': syncat = "QUOTE"
    if( line.startswith("F") ):
        if "M" in line.split()[-1].split("&")[1]:
            elex = 1
            emod = 1
        if re.search( "\d", line.split()[-1].split("&")[1] ):
            elex = 1
            earg = 1
        if( "V" in line.split()[-1].split("&")[1] ): ops["V"] = 1
        if( "Z" in line.split()[-1].split("&")[1] ): ops["Z"] = 1
    if( line.startswith("W") ): w = line.split()[-1]
    if( line.startswith("J") ):
        if "M" in line.split()[-1].split("&")[1]:
            egrm = 1
            emod = 1
        if re.search( "\d", line.split()[-1].split("&")[1] ):
            egrm = 1
            earg = 1
        if "V" in line.split()[-1].split("&")[1]: ops["V"] = 1
        if "Z" in line.split()[-1].split("&")[1]: ops["Z"] = 1
        if line.split()[-1].split("&")[2][0] in "0123456789": ops["Aa"] = 1
        if line.split()[-1].split("&")[3][0] in "0123456789": ops["Ab"] = 1
        if line.split()[-1].split("&")[2] == "M": ops["Ma"] = 1
        if line.split()[-1].split("&")[3] == "M": ops["Mb"] = 1
        if line.split()[-1].split("&")[2] == "C": ops["Ca"] = 1
        if line.split()[-1].split("&")[3] == "C": ops["Cb"] = 1

        if "-g" in syncat or "-g" in basecat: g = 1
        if "-h" in syncat: h = 1
        if "-hO" in syncat: hO = 1

        ops["E"] = elex or egrm
        ops["EnoVHO"] = ops["E"] and (1-ops["V"]) and hO
        ops["EnoVnoHO"] = ops["E"] and (1-ops["V"]) and (1-hO)
        ops["EnoVhNoHO"] = ops["EnoVnoHO"] and h

        # {EnoV} x {"", g, h} x {"", arg, mod} x {"", lex, grm}
        ops["EnoV"] = ops["E"] and (1-ops["V"])
        ops["EnoVg"] = ops["EnoV"] and g
        ops["EnoVh"] = ops["EnoV"] and h
        ops["EnoVarg"] = ops["EnoV"] and earg
        ops["EnoVmod"] = ops["EnoV"] and emod
        ops["EnoVlex"] = ops["EnoV"] and elex
        ops["EnoVgrm"] = ops["EnoV"] and egrm
        ops["EnoVgarg"] = ops["EnoVg"] and earg 
        ops["EnoVgmod"] = ops["EnoVg"] and emod 
        ops["EnoVharg"] = ops["EnoVh"] and earg 
        ops["EnoVhmod"] = ops["EnoVh"] and emod 
        ops["EnoVglex"] = ops["EnoVg"] and elex
        ops["EnoVggrm"] = ops["EnoVg"] and egrm
        ops["EnoVhlex"] = ops["EnoVh"] and elex
        ops["EnoVhgrm"] = ops["EnoVh"] and egrm
        ops["EnoVgarglex"] = ops["EnoVglex"] and earg 
        ops["EnoVgmodlex"] = ops["EnoVglex"] and emod 
        ops["EnoVharglex"] = ops["EnoVhlex"] and earg 
        ops["EnoVhmodlex"] = ops["EnoVhlex"] and emod 
        ops["EnoVgarggrm"] = ops["EnoVggrm"] and earg 
        ops["EnoVgmodgrm"] = ops["EnoVggrm"] and emod 
        ops["EnoVharggrm"] = ops["EnoVhgrm"] and earg 
        ops["EnoVhmodgrm"] = ops["EnoVhgrm"] and emod 
        ops["EnoVarglex"] = ops["EnoVlex"] and earg 
        ops["EnoVmodlex"] = ops["EnoVlex"] and emod 
        ops["EnoVarggrm"] = ops["EnoVgrm"] and earg 
        ops["EnoVmodgrm"] = ops["EnoVgrm"] and emod 

        print(w + " " + syncat + " " + basecat, end="")  
        for op in sorted(ops):
            print(" {}".format(ops[op]), end="")
        print()
