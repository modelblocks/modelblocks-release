import fileinput
import re
import tree
#import model
import quants
#import argparse
import sys
import argparse


parser = argparse.ArgumentParser(description='Print eventuality (semantic) dependencies for GCG tree.')
parser.add_argument('-d', '--debug', default=False, required=False, action='store_true', help='print debug info')
parser.add_argument('-t', '--tree', default=False, required=False, action='store_true', help='enforce tree restriction (for scoring)')
opts = parser.parse_args()



def wid(t):
  if not hasattr(t,'w'):
    t.w = ( 1 if len(t.ch)==0 else wid(t.ch[0]) if len(t.ch)==1 else wid(t.ch[0])+wid(t.ch[1]) )
  return t.w


def econt(t,i=0,a=[],c=[],n=[]):

  if t.c == 'V-iN-gN-lA': t.c = 'V-gN-iN-lA'  # should really be annotated in order of nesting (although #862 still doesn't work)

  os = []
  if t.ch == []: return os

  m = re.search('(-[ghirv][^- {}]*|-[ghirv]\{[^{}]*\})(?=[^ghirv{}]*$)',t.c)
  ns = '' if m==None else m.group(0)
  #ns = re.search('^..*((-[ghirv][^- {}]*|-[ghirv]\{[^{}]*\})*)[^{}]*?$',t.c).group(1)  #re.search('(-[ghirv][^- ]*|-[ghirv]\{[^{}]*\})*(?=[^{}]*$)',t.c).group(0)
  #except:
  #  print ( t.c )
  #  ns = ''
  if opts.debug:
    sys.stderr.write ( ' ' + str(t) + ' i:' + str(i) + ' args:[' + ','.join(a) + '] conjs:[' + ','.join(c) + '] nonlocs:[' + ','.join(n) + '] nonloc-str:<'+ns+'>' + '\n' )

  #### rule T...
  if '-rN' == ns and quants.none_of(t.ch,lambda et: '-r' in et.c):
    if opts.debug: sys.stderr.write ( '   using rule T: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    ns = '-gN'
  if (t.c=='S' or t.c.startswith('S-l')) and quants.some_of(t.ch,lambda et: et.c.startswith('B-aN') and '-l' not in et.c):
    if opts.debug: sys.stderr.write ( '   using rule T: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    a = a + ['-']

  #### rule Ga: start propagating non-local argument down as arg...
#  if len(ns)<5 and quants.none_of(t.ch,lambda et: ns in et.c):
  if '{R' not in ns and '{C' not in ns and quants.none_of(t.ch,lambda et: ns in et.c):
    if opts.debug: sys.stderr.write ( '   using rule Ga: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    if len(n)<1: n=['-']  # kluge for 608, 8911, ..., cats should really be conforming to coling/article.
    a = a + [n[-1]]
    n = n[:-1]
  #### rule V...
  if (t.c.startswith('A-aN') or t.c.startswith('R-aN')) and t.ch[0].c.startswith('L-aN'):
    if opts.debug: sys.stderr.write ( '   using rule V: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    n = [a[0]] if len(a)>0 else ['-']
    ns = '-vN'
    #if len(t.ch)>1 and '-v' not in t.ch[1].c:
    a = ['?']+(a[1:] if len(a)>0 else ['?'])  #a[0] = '-'  #a = ['-']
  #### rule Za...
  if (t.c.startswith('A-aN') or t.c.startswith('R-aN')) and quants.some_of(t.ch,lambda et: (et.ch!=[] and et.c.startswith('N') and '-l' not in et.c) or et.c.endswith('-iN')):
    if opts.debug: sys.stderr.write ( '   using rule Za: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    z = a[0] if a != [] else '-'
    a = a[:-1]
    
  #### terminal...
  if len(t.ch)<1:
    if opts.debug: sys.stderr.write ( 'WARNING: reached terminal\n' )
  #### unary (should be preterminal)...
  elif len(t.ch)<2:
    t.e = '%0.2d' % (i+1)
    p = t if len(t.ch[0].ch)==0 else t.ch[0]
    if p.c == ',': p.c = '-COMMA-'
    if p.c == ';': p.c = '-SEMI-'
    if p.c == ':': p.c = '-COLON-'
    if p.c == '--': p.c = '-DASH-'
    if p.c == '.': p.c = '-STOP-'
    if p.c == '!': p.c = '-BANG-'
    if p.c == '?': p.c = '-QUERY-'
    if p.c == '`': p.c = '-LSQ-'
    if p.c == '\'': p.c = '-RSQ-'
    if p.c == '``': p.c = '-LDQ-'
    if p.c == '\'\'': p.c = '-RDQ-'
    if p.c == 'HYPH': p.c = '-HYPH-'
    if p.ch[0].c == ',': p.ch[0].c = '-COMMA-'
    if p.ch[0].c == ';': p.ch[0].c = '-SEMI-'
    if p.ch[0].c == ':': p.ch[0].c = '-COLON-'
    if p.ch[0].c == '--': p.ch[0].c = '-DASH-'
    if p.ch[0].c == '.': p.ch[0].c = '-STOP-'
    if p.ch[0].c == '!': p.ch[0].c = '-BANG-'
    if p.ch[0].c == '?': p.ch[0].c = '-QUERY-'
    if p.ch[0].c == '`': p.ch[0].c = '-LSQ-'
    if p.ch[0].c == '\'': p.ch[0].c = '-RSQ-'
    if p.ch[0].c == '``': p.ch[0].c = '-LDQ-'
    if p.ch[0].c == '\'\'': p.ch[0].c = '-RDQ-'
    os += [t.e+',0,'+p.c+':'+p.ch[0].c]
    if t.c.startswith('Q') or '-a{' in t.c: a.reverse()  ## implements subj-aux inversion.
    l = a+c+n if c == [] else c
    os += [t.e+'r,'+str(j+1)+','+l[j] for j in range(0,len(l))]  # if l[j]!='-']
    if opts.debug: sys.stderr.write ( '   ' + ' '.join(os) + '\n' )
    if opts.debug and len(re.findall('-(.\{[^\{\}]*\}|.[A-Z])',p.c))-1 != len(a+c+n):
      print ( '   WARNING: cat ' + p.c + ' (' + ' '.join(re.findall('-(.\{[^\{\}]*\}|.[A-Z])',p.c)) + ') for word "' + p.ch[0].c + '" has non-standard number of arguments: ' + ' '.join(a+c+n) )
  #### rule Fc...
  elif ('-gN}' in t.ch[0].c or '-iN}' in t.ch[0].c) and '-lA' in t.ch[1].c:
    if opts.debug: sys.stderr.write ( '   using rule Fc: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    # if tough construction, pass subj as nonloc...
    if t.ch[0].c.endswith('I-aN-gN}') or 'I-aN-gN}-l' in t.ch[0].c:
      os += econt ( t.ch[1], i+wid(t.ch[0]), ['*'],               c,  a )
    else:
      os += econt ( t.ch[1], i+wid(t.ch[0]), [],                  c, (n if ns in t.ch[1].c else []) + ['-'] )
    os +=   econt ( t.ch[0], i,              a + [t.ch[1].e+'s'], c,  n if ns in t.ch[0].c else [] )
    t.e = t.ch[0].e
  #### rule Fa/Fb...  # second disjunct below is stupid; null rel / nom clause should really be N-b(V-gN)
  elif ('-lN' in t.ch[0].c or '-lN' in t.ch[1].c) and '-g' in t.ch[1].c:   #('-g' not in t.c or ns not in t.ch[1].c) and '-g' in t.ch[1].c:   #and '-lN' in t.ch[0].c
    m = re.search('(-[ghirv][^- {}]*|-[ghirv]\{[^{}]*\})+(?=[^{}]*$)',t.ch[1].c)
    ns1 = '' if m==None else m.group(0)
    #ns1 = re.search('^..*?((-[ghirv][^- {}]*|-[ghirv]\{[^{}]*\})*)[^{}]*?$',t.ch[1].c).group(1)  #re.search('(-[ghirv][^- ]*|-[ghirv]\{[^{}]*\})*(?=[^{}]*$)',t.ch[1].c).group(0)
    #### rule Fa...
    if '{R' not in ns1 and '{C' not in ns1: #len(ns1)<5:
      if opts.debug: sys.stderr.write ( '   using rule Fa: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
      os += econt ( t.ch[0], i,              re.findall('(?<=^.)-(?=a)',t.ch[0].c), c, (n if ns in t.ch[0].c else []) + (['*'] if '-i' not in t.c and '-i' in t.ch[0].c else []) )
      os += econt ( t.ch[1], i+wid(t.ch[0]), a                                    , c, (n if ns in t.ch[1].c else []) + [t.ch[0].e+'s'] )
      t.e = t.ch[0].e if '-lN' in t.ch[1].c else t.ch[1].e
    #### rule Fb...
    else:
      if opts.debug: sys.stderr.write ( '   using rule Fb: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
      os += econt ( t.ch[1], i+wid(t.ch[0]), a,         c,  n if ns in t.ch[1].c else [] )
      os += econt ( t.ch[0], i,              t.ch[1].n, c, (n if ns in t.ch[0].c else []) + (['*'] if '-i' not in t.c and '-i' in t.ch[0].c else []) )
      t.e = t.ch[0].e if '-lN' in t.ch[1].c else t.ch[1].e
  #### rule Ra/Rb...
  elif '-r' not in t.c and '-r' in t.ch[1].c:
    if opts.debug: sys.stderr.write ( '   using rule Ra/b: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += econt ( t.ch[0], i,              a,  c,  n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), [], c, (n if ns in t.ch[1].c else []) + [t.ch[0].e+'r'] )
    t.e = t.ch[0].e
  #### rule Rbogus -- sentence-initial 'when' should really have been annotated R-aN-bV...
  elif '-r' not in t.c and '-r' in t.ch[0].c:
    os += econt ( t.ch[1], i+wid(t.ch[0]), a,  c,  n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[0], i,              [], c, (n if ns in t.ch[0].c else []) + [t.ch[1].e+'r'] )
    t.e = t.ch[1].e
  #### rule Rbogus2 -- parenthetical should really have been annotated something else...
  elif '-gS' not in t.c and 'V-gS-lN' in t.ch[0].c:
    os += econt ( t.ch[1], i+wid(t.ch[0]), a,  c,  n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[0], i,              [], c, (n if ns in t.ch[0].c else []) + [t.ch[1].e+'r'] )
    t.e = t.ch[1].e
  #### rule Ha/Hb...
  elif ('-h' not in t.c and '-h' in t.ch[0].c) or ('-g' not in t.c and '-g' in t.ch[0].c):  # backward gap should really have been annotated '-h'.
    ns0 = re.search('(-[ghirv][^- ]*|-[ghirv]\{[^{}]*\})*(?=[^{}]*$)',t.ch[0].c).group(0)
    #### rule Ha...
    if '{R' not in ns0 and '{C' not in ns0: #len(ns0)<5:
      if opts.debug: sys.stderr.write ( '   using rule Ha: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
      os += econt ( t.ch[1], i+wid(t.ch[0]), re.findall('(?<=^.)-(?=a)',t.ch[1].c), c,  n if ns in t.ch[1].c else [] )
      os += econt ( t.ch[0], i             , a                                    , c, (n if ns in t.ch[0].c else []) + [t.ch[1].e+'s'] )
      t.e = t.ch[0].e
    #### rule Hb...
    else:
      if opts.debug: sys.stderr.write ( '   using rule Hb: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
      os += econt ( t.ch[0], i,              a,         c, n if ns in t.ch[0].c else [] )
      os += econt ( t.ch[1], i+wid(t.ch[0]), t.ch[0].n, c, n if ns in t.ch[1].c else [] )
      t.e = t.ch[0].e
  #### rule Aa...
  elif '-lA' in t.ch[0].c:
    if opts.debug: sys.stderr.write ( '   using rule Aa/c/e/g: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
#    os += econt ( t.ch[0], i,              ['??'] if '-a' in t.ch[0].c else [], c, n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[0], i,              re.findall('(?<=^.)-(?=a)',t.ch[0].c),             c, n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), a+[t.ch[0].e+'s'] if t.ch[0].c != 'Ne-lA' else a, c, n if ns in t.ch[1].c else [] )
    t.e = t.ch[1].e
  #### rule Ab...
  elif '-lA' in t.ch[1].c:
    if opts.debug: sys.stderr.write ( '   using rule Ab/d/f/h: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
#    a1 = ['-']
#    # obtain reft for complex arg...
#    if not opts.tree and '-a' in t.ch[1].c:
#      os1 = econt ( t.ch[0], i, a+['-'], c, n if ns in t.ch[0].c else [] )
#      lv = [o.split(',')[2] for o in os1 if str(t.ch[0].e)+',1,' in o]
#      a1 = [lv[0]] if len(lv)>0 else ['-']
#    os += econt ( t.ch[1], i+wid(t.ch[0]), a1 if '-a' in t.ch[1].c else [],  c, n if ns in t.ch[1].c else [] )
#    os += econt ( t.ch[1], i+wid(t.ch[0]), ['??'] if '-a' in t.ch[1].c else [], c, n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), re.findall('(?<=^.)-(?=a)',t.ch[1].c), c, n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[0], i,              a+[t.ch[1].e+'s'] if t.ch[1].c != 'Ne-lA' else a,                     c, n if ns in t.ch[0].c else [] )
    t.e = t.ch[0].e
  #### rule Ma...
  elif '-lM' in t.ch[0].c:
    if opts.debug: sys.stderr.write ( '   using rule Ma/c/e/g: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += econt ( t.ch[1], i+wid(t.ch[0]), a,               c,  n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[0], i,              [t.ch[1].e+'r'], [], n if ns in t.ch[0].c else [] )
    t.e = t.ch[1].e
  #### rule Mb...
  elif '-lM' in t.ch[1].c:
    if opts.debug: sys.stderr.write ( '   using rule Mb/d/f/h: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += econt ( t.ch[0], i,              a,               c,  n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), [t.ch[0].e+'r'], [], n if ns in t.ch[1].c else [] )
    t.e = t.ch[0].e
  #### rule Ca/Cc...
  elif '-lC' in t.ch[0].c:
    if opts.debug: sys.stderr.write ( '   using rule Ca: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += econt ( t.ch[0], i,              a, [],                n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), a, c+[t.ch[0].e+'c'], n if ns in t.ch[1].c else [] )
    t.e = t.ch[1].e
  #### rule Cb...
  elif '-lC' in t.ch[1].c:
    if opts.debug: sys.stderr.write ( '   using rule Cb: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += econt ( t.ch[1], i+wid(t.ch[0]), a, [],                n if ns in t.ch[1].c else [] )
    os += econt ( t.ch[0], i,              a, c+[t.ch[1].e+'c'], n if ns in t.ch[0].c else [] )
    t.e = t.ch[0].e
  #### fail...
  else:
    sys.stderr.write ( 'WARNING: assuming conjunction for: ' + t.c + ' ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    os += econt ( t.ch[0], i,              a, c, n if ns in t.ch[0].c else [] )
    os += econt ( t.ch[1], i+wid(t.ch[0]), a, c, n if ns in t.ch[1].c else [] )
    t.e = t.ch[0].e

  #### rules Ac-h,Mc-h: propagate non-local modifier up...
  t.n = []
  if len(t.ch)>1 and ns in t.ch[0].c: t.n += t.ch[0].n
  if len(t.ch)>1 and ns in t.ch[1].c: t.n += t.ch[1].n

  #### rule Gb: start propagating non-local modifier up...
#  if len(ns)>=5 and quants.none_of(t.ch,lambda et: ns in et.c):
  if ('{R' in ns or '{C' in ns) and quants.none_of(t.ch,lambda et: ns in et.c):
    if opts.debug: sys.stderr.write ( '   using rule Gb: ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    t.n += [t.e+'r']

  #### rule Za...
  if (t.c.startswith('A-aN') or t.c.startswith('R-aN')) and quants.some_of(t.ch,lambda et: (et.ch!=[] and et.c.startswith('N') and '-l' not in et.c) or et.c.endswith('-iN')):
    if opts.debug: sys.stderr.write ( '   using rule Za (again): ' + ' '.join([t.c]+[et.c for et in t.ch]) + '\n' )
    os += [t.e+'r,'+('~' if t.c.startswith('A-aN-x') else '=' if t.c.startswith('A-aN') else '@')+','+z] #if z != '-' else []

  return os


ln = 0
for line in sys.stdin:
    ln += 1
    if opts.debug: sys.stderr.write ( str(ln) + '\n' )
    t = tree.Tree()
    try:
      t.read(line)
    except:
      sys.stderr.write ( 'Error reading tree in line ' + str(ln) + ': ' + line + '\n' )
    #print ( str(t) )
    os = sorted(econt(t))
    os = ['00,1,'+t.e] + os if hasattr(t,'e') else []
    print ( ' '.join(os) )


