import sys
import cuegraphcontexts
import pdb

## for each article until eof...
article_number = 0
eof = False
while ( not eof ):

  ## for each sentence in article...
  G = cuegraphcontexts.CueGraph()
  for linenum,sent in enumerate(sys.stdin):
    if '!article' in sent: break
    ## for each dep...
    for dep in sent.split():
      if len(dep.split(',')) != 3:
        sys.stderr.write('WARNING: Improperly formed dependency: '+dep+'\n')
        continue
      if dep.startswith('00,1,'):
        continue
      src,lbl,dst = dep.split(',')
      if src != '-' and len(src)<=3: src = str(linenum+1)+src
      if dst != '-' and len(dst)<=3: dst = str(linenum+1)+dst
      G.add(src,lbl,dst)
  ## if all lines read, end article loop...
  else: eof = True

  ## calculate contexts in article...
  C = cuegraphcontexts.CueGraphContexts(G)
  for v in sorted(C):
    if not ":" in v:
      print ( str(article_number)+'-'+v + ' ' + ' '.join(C[v]) )
  if article_number>0: print ( )
  article_number += 1





