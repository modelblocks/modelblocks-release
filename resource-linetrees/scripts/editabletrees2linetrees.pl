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

#!/usr/bin/env perl

use strict;
my $num_parens=0;
my $tree = "";

while(<STDIN>){
  chomp;
  ## Get rid of leading spaces
  s/^\s*//;

  if($_ eq ""){
    next;
  }
  if(m/^\*/){
    ## get rid of brown headers
    next;
  }

  # Accumulate tree lines until parens match
  $tree .= "$_ ";
  my @left_parens = m/(\()/g;
  my @right_parens = m/(\))/g;
  $num_parens += ($#left_parens + 1);
  $num_parens -= ($#right_parens + 1);
  if($num_parens == 0){
    ## Get rid of extra parentheses around the tree
    $tree =~ s/^\s*\((?:TOP)?\s*\((.*)\s*\)\s*\)\s*$/(\1)/;

    ## Get rid of extra spaces
    $tree =~ s/ *\)/\)/g;
    $tree =~ s/ +/ /g;
    $tree =~ s/ $//;

    print "$tree\n";
    $tree = "";
  }
#  print $_;
}
