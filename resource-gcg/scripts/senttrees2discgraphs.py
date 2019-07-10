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
import tree
import gcgtree
import semcuegraph

RELABEL = False

for a in sys.argv:
  if a=='-d':
    gcgtree.VERBOSE = True
    semcuegraph.VERBOSE = True
  if a=='-r':
    RELABEL = True

################################################################################

discctr = 0
sentctr = 0
G       = semcuegraph.SemCueGraph( )

for line in sys.stdin:

  if '!ARTICLE' in line:
    if discctr>0: print( str(G) )
    sentctr = 0
    G = semcuegraph.SemCueGraph( )
    discctr += 1

  else:
    if RELABEL: t = gcgtree.GCGTree( line )
    else:
      t = tree.Tree( )
      t.read( line )
    G.add( t, ('0' if sentctr<10 else '') + str(sentctr) )
    sentctr += 1

print( str(G) )

