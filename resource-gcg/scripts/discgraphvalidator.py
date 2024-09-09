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

import sys
import os
import collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import discgraph

###############################################################################

## For each discourse graph...
for discnum,line in enumerate( sys.stdin ):

  DiscTitle = sorted([ asc  for asc in line.split(' ')  if ',0,' in asc and asc.startswith('000') ])
  print(            '#DISCOURSE ' + str(discnum+1) + '... (' + ' '.join(DiscTitle) + ')' )
  sys.stderr.write( '#DISCOURSE ' + str(discnum+1) + '... (' + ' '.join(DiscTitle) + ')\n' )

  ## Read in line and check...
  line = line.rstrip()
  D = discgraph.DiscGraph( line )
  if D.check():
    D.checkMultipleOutscopers() 

