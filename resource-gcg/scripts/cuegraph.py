import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import gcgtree

VERBOSE = False


################################################################################

class CueGraph( dict ):

  def __init__( G, s ):
    G = { }
    for dep in s.split():
      x,l,y = re.match('([^,]*),([^,]*),(.*)',dep).groups()
      G[x,l]=y

  def __str__( G ):
    return ' '.join( [ ','.join( (x,l,G[x,l]) ) for x,l in G ] )

  def rename( G, xNew, xOld ):
    if xOld != xNew:
      for z,l in G.keys():
        if G[z,l] == xOld: G[z,l] = xNew     ## replace old destination with new
        if z == xOld:                        ## replace old source with new
          if (xNew,l) not in G:
            G[xNew,l] = G[xOld,l]
            del G[xOld,l]
      for z,l in G.keys():
        if z == xOld and (xNew,l) in G: G.rename( G[xNew,l], G[xOld,l] )

  def result( G, l, x ):                     ## (f_l x)
    if (x,l) not in G:  G[x,l] = x+l         ## if dest is new, name it after source and label
    return G[x,l]

  def equate( G, y, l, x ):                  ## y = (f_l x)
    if (x,l) in G: G.rename( y, G[x,l] )     ## if source and label exist, rename
    else:          G[x,l] = y                ## otherwise, add to dict



