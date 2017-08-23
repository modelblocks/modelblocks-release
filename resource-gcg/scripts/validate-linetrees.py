import sys
import os
import re
import collections
import tree
import morph


def validate( t, W, L ):

  ## apply morph rules...
  if len( t.ch )==1 and len( t.ch[0].ch )==0:
    s = morph.getLemma( t.c, t.ch[0].c )
    cat,base = s.split( ':' )
    W[ re.sub( '^(.)[^:]*', '\\1', re.sub('-[lx].*','',t.c) ) + ':' + t.ch[0].c.lower() ] += 1
    if base!=t.ch[0].c: L[ re.sub( '^(.)[^:]*', '\\1', s ) ] = re.sub('-l.','',t.c) + ':' + t.ch[0].c.lower()

  ## recurse...
  for st in t.ch:
    validate( st, W, L )


W = collections.defaultdict( int )
L = collections.defaultdict( int )
for linenum,line in enumerate( sys.stdin ):
  line = re.sub( ':', '!colon!', line )
  t = tree.Tree( )
  t.read( line )
  validate( t, W, L )
  if linenum%10000==0: sys.stderr.write( 'Reading line ' + str(linenum) + '...\n' )

for lemma in sorted(L):
  if lemma not in W:
    print( 'no natural occurrences: ' + lemma + ' (from ' + L[lemma] + ')' )


