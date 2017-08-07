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

import re

def getLemma( c, w ):  
  s = re.sub('-l.','',c) + ':' + w.lower()
  while( '-x' in s ):
    s1 = re.sub( '^.(\\S*?)-x.%:(\\S*)%(\\S*)\|(\\S*)%(\\S*):([^% ]*)%([^-: ]*)([^: ]*):\\2(\\S*)\\3', '\\4\\1\\5\\8:\\6\\9\\7', s )
    if s1==s: s1 = re.sub( '^.(\\S*?)-x.%(\\S*)\|([^% ]*)%([^-: ]*)([^: ]*):(\\S*)\\2', '\\3\\1\\5:\\6\\4', s )
    if s1==s: s1 = re.sub( '-x', '', s )
    s = s1
  s = re.sub( '([AB])-aD-bO', '\\1-aN-bN', s )
  s = re.sub( '([AB])-aD', '\\1-aN', s )
  return s

