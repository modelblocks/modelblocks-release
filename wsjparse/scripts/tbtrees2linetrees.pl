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
  chomp;
  ## Get rid of leading spaces
  s/^\s*//;

  if($_ eq ""){
    next;
  }

#  s/PRT\|//g;

#  ## Get rid of the dash in *-UNF categories
#  s/-UNF/UNF/g;
#  ## Get rid of everything after the dash (eg. VP-PRD but not -DFL-)
#  s/(\([^ -]+)-[^ ]+/\1/g;
#  ## Escape some difficult terminal symbols to non-word letter combinations
#  s/\\\]/SUBSTITUTERIGHTBRACKET/g;

##  s/\\\[/SUBSTITUTELEFTBRACKET/g;
##  s/\\\+/SUBSTITUTEPLUS/g;
##  s/\.\)/SUBSTITUTEDOT)/g;

##  # Eliminate empty categories (and the unary branch above them) and change right-recursive branch above that to reflect deletion
##  s/\(([^ ]+) \(\-NONE\- [^ ]+\) \)//;
##  # Eliminate all remaining empty categories (without unary branches above)
##  s/\(\-NONE\- [^ ]+\)//;

#  ## Remove cues that give away edited-ness
#  s/\(RM \(\-DFL\- \\\[\) \)//g;
#  s/\(IP \(\-DFL\- \\\+\) \)//g;
#  s/\(-DFL- E_S\)//g;
#  s/\(-DFL- N_S\)//g;
#  s/\(RS \(\-DFL\- SUBSTITUTERIGHTBRACKET\) \)//g;

#  # Remove punctuation
#  s/\(\. \.\)//g;
#  s/\(\. \?\)//g;
#  s/\(, ,\)//g;

  # Accumulate tree lines until parens match
  $tree .= "$_ ";
  my @left_parens = m/(\()/g;
  my @right_parens = m/(\))/g;
  $num_parens += ($#left_parens + 1);
  $num_parens -= ($#right_parens + 1);
  if($num_parens == 0){
    ## Get rid of extra parentheses around the tree
    $tree =~ s/^\s*\(\s*\((.*)\s*\)\s*\)\s*$/(\1)/;
##    print substr($tree, 2, length($tree) - 4) . "\n"; # "$tree\n";

#    ## Now that tree is on one line, eliminate / push upward many kinds of empty category crapola...
#    $tree =~ s/\(([^ ]+) +\(([^ ]+) +\(\-NONE\- +[^ ]+ *\) *\)/\(\1startingwithempty\2/g;
#    $tree =~ s/\(([^ ]+) +\(\-NONE\- +[^ ]+ *\) *\)/\(empty\1 -NONE-\) /g;
#    $tree =~ s/\(([^ ]+) +\(\-NONE\- +[^ ]+ *\)/\(\1startingwithempty/g;
#    $tree =~ s/\(([^ ]+)startingwithempty[^ ]* +\(empty[^ ]+ +\-NONE\- *\) *\)/\(empty\1 -NONE-\)/g;
#    $tree =~ s/\(\-NONE\- [^ ]+ *\)/\(emptysomething -NONE-\)/g;

##    $tree =~ s/ +\(([^ ]+) +\(\-NONE\- +[^ ]+\) *\)/endingwithempty\1/g;

    print "$tree\n";
    $tree = "";
  }
#  print $_;
}
