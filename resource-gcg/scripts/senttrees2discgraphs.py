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
  if a=='-e':
    semcuegraph.EQN_DEFAULTS = True
  if a=='-r':
    RELABEL = True

################################################################################

discctr = 0
finished = False

## For each discourse...
while not finished:

  ## Define discourse graph...
  G = semcuegraph.SemCueGraph( )
  sentctr = 0

  ## For each sentence...
  for line in sys.stdin:

    if '!ARTICLE' in line: break

    if sentctr == 0:
      discctr += 1
      sys.stderr.write( 'Discourse ' + str(discctr) + ': ' + line )

    ## Initialize new tree with or without tree-lengthening...
    if RELABEL:
      t = gcgtree.GCGTree( line )
    else:
      t = tree.Tree( )
      t.read( line )
    ## Add tree to discourse graph...
    G.add( t, ('0' if sentctr<10 else '') + str(sentctr) )
    sentctr += 1

  else: finished = True

  ## If discourse contained any sentences, finalize and print graph...
  if sentctr>0:
    G.finalize()
    print( str(G) )


