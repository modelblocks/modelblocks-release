import sys
import sets
import collections

print( '\\documentclass[tikz]{standalone}' )
print( '\\tikzset{>=latex,->}' )
print( '\\tikzstyle{reft}=[circle,minimum width=2.5ex,inner sep=0pt,draw]' )
print( '\\tikzstyle{pred}=[node distance=20mm,anchor=mid]' )
print( '\\begin{document}' )

## For each discourse graph...
for line in sys.stdin:

  line = line.rstrip()

#  print( 'digraph G {' )   ## dot
  print( '\\begin{tikzpicture}[x=1cm,y=3cm,scale=3]' )

  Nodes = sets.Set()
  Ctrs  = collections.defaultdict( int )

  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split( ',', 2 )
 
#    print( '"' + src + '" -> "' + dst + '" [label="' + lbl + '", color=' + ('black' if lbl in '0123456789' else 'magenta' if lbl=='s' else 'blue') + '];' )   #, constraint=false];' )  ## dot

    for x in [src] if lbl=='0' else [src,dst]:
      if x not in Nodes:

        def hoff( x, ctr ):
          return str( (4 if len(x)==5 else 2) + (2 if x[-1]=='r' else 4) + (0 if len(x)==5 else -ctr if x[4] in 'abu' else ctr) )
        def voff( x, ctr ):
          return str( 4 if len(x)==5 else (4 - ctr) if x[4] in 'abu' else (4 + ctr) )

        if len(x)>5 and x[-1]!='r': Ctrs[ x[0:4] ] += 1
        ctr = Ctrs[ x[0:4] ]

        Nodes.add( x )
#        if   x[4]=='a' and len(x)>7 and x[7]=='Q':  print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.4) [reft]{' + x[4:] + '};' )
#        elif x[4]=='a':                             print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.2) [reft]{' + x[4:] + '};' )
#        elif x[4]=='b' and len(x)>7 and x[7]=='Q':  print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.5) [reft]{' + x[4:] + '};' )
#        elif x[4]=='b':                             print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.3) [reft]{' + x[4:] + '};' )
#        elif x[4]=='u':                             print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.3) [reft]{' + x[4:] + '};' )
#        elif x[4:]=='s':                            print( '\\node(x' + x + ') at (' + x[2:4] +                                '.8,-' + x[0:2] + '.4) [reft]{' + x[4:] + '};\n' +
        if   x[4:]=='s':                            print( '\\node(x' + x + ') at (' + x[2:4] +                                '.8,-' + x[0:2] + '.4) [reft]{' + x[4:] + '};\n' +
                                                           '\\node(w' + x + ')[above of=x' + x + ',pred]{' + x[0:4] + '};\n' +
                                                           '\\draw(w' + x + ') -- node[left]{S} (x' + x + ');' )
#        elif x[4:]=='r':                            print( '\\node(x' + x + ') at (' + x[2:4] +                                '.6,-' + x[0:2] + '.4) [reft]{' + x[4:] + '};' )
#        elif x[4:6]=='sE':                          print( '\\node(x' + x + ') at (' + x[2:4] + ('.2' if x[-1]=='r' else '.4') + ',-' + x[0:2] + '.4) [reft]{' + x[4:] + '};' )
#        elif x[4:6]=='sQ':                          print( '\\node(x' + x + ') at (' + x[2:4] + ('.4' if x[-1]=='r' else '.6') + ',-' + x[0:2] + '.5) [reft]{' + x[4:] + '};' )
#        elif x[4:6]=='sR':                          print( '\\node(x' + x + ') at (' + x[2:4] + ('.6' if x[-1]=='r' else '.8') + ',-' + x[0:2] + '.6) [reft]{' + x[4:] + '};' )
#        else:                                       print( '\\node(x' + x + ') at (' + x[2:4] +                                '.0,-' + x[0:2] + '.4) [reft]{' + x[4:] + '};' )
        else:                                       print( '\\node(x' + x + ') at (' + x[2:4] + '.' + hoff(x,ctr) + ',-' + x[0:2] + '.' + voff(x,ctr) + ') [reft]{' + x[4:] + '};' ) 

    if lbl == '0':
      print( '\\node(k' + src[0:4] + ')[below of=x' + src + ',pred] {' + dst + '};' )
      print( '\\draw(x' + src + ') -- node[left]{0} (k' + src[0:4] + ');' )
    elif lbl == 's':
      print( '\\draw[orange](x' + src + ') to [bend ' + ('left' if src[0:4]<dst[0:4] else 'right') + ',near start] node[above]{' + lbl + '} (x' + dst + ');' )
    elif lbl >= '1' and lbl <= '9':
      print( '\\draw(x' + src + ') to [bend ' + ('right' if src[0:4]<=dst[0:4] else 'left') + ',near start] node[below]{' + lbl + '} (x' + dst + ');' )
    elif lbl == 'r' and src[0:4]==dst[0:4]:
      print( '\\draw[blue](x' + src + ') -- node[below]{' + lbl + '} (x' + dst + ');' )
    else:
      print( '\\draw[blue](x' + src + ') to [bend ' + ('right' if src[0:4]<dst[0:4] else 'left') + ',near start] node[below]{' + lbl + '} (x' + dst + ');' )

  print( '\\end{tikzpicture}' )

#  print( '{ ordering=out; x->"' + '"[style="invis"]; x->"'.join([ x for x in Nodes if ':' not in x ]) + '"[style="invis"]; }' )  ## dot
#  print( '{ rank=same; "' + '"; "'.join(sorted([ x for x in Nodes if ':' in x ])) + '"; }' )   ## dot

#  print( '}' )  ## dot

print( '\\end{document}' )  



