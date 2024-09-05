##############################################################################
##                                                                           ##
## This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                           ##
##    ModelBlocks is free software: you can redistribute it and/or modify    ##
##    it under the terms of the GNU General Public License as published by   ##
##    the Free Software Foundation, either version 3 of the License, or      ##
##    (at your option) any later version.                                    ##
##                                                                           ##
##    ModelBlocks is distributed in the hope that it will be useful,         ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License for more details.                           ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   ##
##                                                                           ##
###############################################################################

import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import gcgtree

################################################################################

class CueGraph( dict ):

  def __init__( G, s ):
    G = { }
    for dep in s.split():
      x,l,y = re.match('([^,]*),([^,]*),(.*)',dep).groups()
      G[x,l]=y

  def __str__( G ):
    return ' '.join( [ ','.join( (x,l,G[x,l]) ) for x,l in sorted(G) ] )

  def rename( G, xNew, xOld ):
    if xOld != xNew:
      for z,l in list( G.keys() ):
        if G[z,l] == xOld: G[z,l] = xNew     ## replace old destination with new
        if z == xOld:                        ## replace old source with new
          if (xNew,l) not in G:
            G[xNew,l] = G[xOld,l]
            del G[xOld,l]
      for z,l in list( G.keys() ):
        if z == xOld and (xNew,l) in G:
          G.rename( G[xNew,l], G[xOld,l] )
          del G[xOld,l]

  def result( G, l, x ):                     ## (f_l x)
    if (x,l) not in G:  G[x,l] = x+l         ## if dest is new, name it after source and label
    return G[x,l]

  def equate( G, y, l, x ):                  ## y = (f_l x)
    if (x,l) in G: G.rename( y, G[x,l] )     ## if source and label exist, rename
    else:          G[x,l] = y                ## otherwise, add to dict



