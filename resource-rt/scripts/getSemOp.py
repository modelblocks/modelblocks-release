import os, sys, re
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
#import tree

print( 'word opAa opAb opMa opMb opCa opCb opE opV opZ' )
for line in sys.stdin:
    if( line.startswith('F') ):
        opAa,opAb,opMa,opMb,opCa,opCb,opE,opV,opZ = '0','0','0','0','0','0','0','0','0'
#        if( line.split()[-1].split('&')[1] != '' and line.split()[-1].split('&')[1][0] in 'M0123456789' ): opE='1'
        if( 'M' in line.split()[-1].split('&')[1] or re.search( '\d', line.split()[-1].split('&')[1] ) ): opE='1'
        if( 'V' in line.split()[-1].split('&')[1] ): opV='1'
        if( 'Z' in line.split()[-1].split('&')[1] ): opZ='1'
    if( line.startswith('W') ): w = line.split()[-1]
    if( line.startswith('J') ):
#        if( line.split()[-1].split('&')[1] != '' and line.split()[-1].split('&')[1][0] in 'M0123456789' ): opE='1'
        if( 'M' in line.split()[-1].split('&')[1] or re.search( '\d', line.split()[-1].split('&')[1] ) ): opE='1'
        if( 'V' in line.split()[-1].split('&')[1] ): opV='1'
        if( 'Z' in line.split()[-1].split('&')[1] ): opZ='1'
        if( line.split()[-1].split('&')[2][0] in '0123456789' ): opAa='1'
        if( line.split()[-1].split('&')[3][0] in '0123456789' ): opAb='1'
        if( line.split()[-1].split('&')[2] == 'M' ): opMa='1'
        if( line.split()[-1].split('&')[3] == 'M' ): opMb='1'
        if( line.split()[-1].split('&')[2] == 'C' ): opCa='1'
        if( line.split()[-1].split('&')[3] == 'C' ): opCb='1'
        print( w + ' ' + opAa + ' ' + opAb + ' ' + opMa + ' ' + opMb + ' ' + opCa + ' ' + opCb + ' ' + opE + ' ' + opV + ' ' + opZ )

#    if (line.strip() !='') and (line.strip()[0] != '%'):
#        T = tree.Tree()
#        T.read(line)
#        for word, cat in zip(T.words(), T.syncats()):
#            print(word + " " + cat)
