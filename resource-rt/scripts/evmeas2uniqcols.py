import sys

for linenum,line in enumerate(sys.stdin):
    if linenum==0:
        measures = line.split()
        for i in range( 0, len(measures) ):
            ctr = 1
            for j in range( i+1, len(measures) ):
                #sys.stderr.write( measures[i]  + ' ?? ' + measures[j] + '\n' )
                if measures[i]==measures[j]:
                    #if ctr==1: measures[i]+=str(ctr)
                    ctr += 1
                    measures[j]+=str(ctr)
        print( ' '.join(measures) )
    else: sys.stdout.write( line )

