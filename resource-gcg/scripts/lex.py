import sys
import re

def getFn( c, w ):
  c = re.sub('(-l.|-[mnstuw][0-9]+r?|-yQ)','',c)
  s = c + ':' + w.lower()
  expr = re.sub( '-x.*:', ':', s )
  for xrule in re.split( '-x', c )[1:] :
    ## If morphological macro...
    # (none yet)
    ## If 2-variable compositional lex rule, apply...
    m = re.search( '(.*)%(.*)%(.*)\|(.*)%(.*)%(.*)', xrule )
    if m is not None:
      expr = re.sub( '\\b'+m.group(1)+'([^ ]*)'+m.group(2)+'([^ ]*)'+m.group(3)+'$', m.group(4)+'\\1'+m.group(5)+'\\2'+m.group(6), expr )
      continue
    ## If 1-variable compositional lex rule, apply...
    m = re.search( '(.*)%(.*)\|(.*)%(.*)', xrule )
    if m is not None:
      expr = re.sub( '\\b'+m.group(1)+'([^ ]*)'+m.group(2)+'$', m.group(3)+'\\1'+m.group(4), expr )
      continue
    ## If wildcard-deleted compositional lex rule, apply...
    m = re.search( '.*%.*\|(.*)', xrule )
    if m is not None:
      expr = m.group(1)
      continue
#    ## If non-morphological macro, cram onto function...
#    for m in re.findall( '^[A-Z0-9]+$', xrule ):
#      expr = expr[0] + xrule + expr[1:]
  expr = re.sub( '([AB])-aD-bO', '\\1-aN-bN', expr )
  expr = re.sub( '([AB])-aD', '\\1-aN', expr )
  return expr



