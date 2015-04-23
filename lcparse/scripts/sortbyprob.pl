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


#### new memory-stingy algo

while (<>) {
  if ( ($c,$v,$p) = ($_ =~ /^(.*) : (.*) = (.*)$/) ) {
    if ( $c ne $cond ) {
      foreach $pr (sort{$b<=>$a} keys %CPT) {
        foreach $val (sort keys %{$CPT{$pr}}) {
          print "$cond : $val = $pr\n";
          #print "$cond : $val = ".(1.0-$pr)."\n";
        }
      }
      $cond = $c;
      %CPT={};
    }
    $CPT{$p}{$v}=1;
    #$CPT{1.0-$p}{$v}=1;
  }
}

foreach $pr (sort{$b<=>$a} keys %CPT) {
  foreach $val (sort keys %{$CPT{$pr}}) {
    print "$cond : $val = $pr\n";
    #print "$cond : $val = ".(1.0-$pr)."\n";
  }
}



#### old memory-lavish algo

# while (<>) {
#   if ( $_ =~ /^(.*) : (.*) = (.*)$/ ) {
#     $CPT{$1}{1.0-$3}{$2}=1;
#   }
# }

# foreach $c (sort keys %CPT) {
#   foreach $p (sort keys %{$CPT{$c}}) {
#     foreach $v (sort keys %{$CPT{$c}{$p}}) {
#       print "$c : $v = ".(1.0-$p)."\n";
#     }
#   }
# }
