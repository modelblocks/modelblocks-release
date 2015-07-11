###############################################################################
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

#!/usr/bin/perl

use strict;
my $num_parens=0;
my $tree = "";

while(<STDIN>){
  while ( s/(<[^>]*)\(([^>]*>)/\1\{\2/g ) { }
  while ( s/(<[^>]*)\)([^>]*>)/\1\}\2/g ) { }
  s/<[^ ]* ([^ ]*) [^ ]* [^ ]* ([^ ]*) [^ ]*>/\1 \2/g;
  s/<.*? (.*?) .*? .*?>/\1/g;
  s/[\[\]]//g;
  s/ \)/\)/g;
  s/ *$//;
  if ($_ !~ /^ID/) { print $_; }
}
