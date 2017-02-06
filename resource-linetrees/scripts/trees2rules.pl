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

use Getopt::Std;

getopts("dl");

$DEBUG = 0;
if ($opt_d) {
  $DEBUG = 1;
}

sub debug {
  if ($DEBUG) {
    $msg = $_[1];
    print stderr $_[0] , " " , $msg, "\n";
  }
}

## for each tree...
while ( <> ) {
  s/[\[\]]/!/g; # change bracks to exclams 

  if ($opt_l) { print $_; }

  ## if root...
  if ( ($c) = ($_ =~ /^\(([^ ]*) /) ) {
    print "Cr : $c\n";
  }

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    $_ =~ s/\(([^\(\)]*)\)/\^\1\^/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug($step++, "   $_");

     ## if terminal branch...
     if ( ($c,$x) = ($_ =~ /\^([^ ]*) ([^ \(\)]*)\^/) ) {
       print "CC $c : \n";
       print "X $c : $x\n";
     }
     ## if nonterminal (any n-ary) branch...
     elsif ( ($c,$cc) = ($_ =~ /\^([^ ]*)(.*<[^ ]*) [^<]*\^/) ) {
       @A = split ( / [^<]*</, $cc );
       print "CC $c :@A\n";
     }

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    $_ =~ s/\^(.*)\^/<\1>/;
  }
  ## finish up...
  $_ =~ s/</[/;
  $_ =~ s/>/]/;
  ## translate to parens again...
  $_ =~ s/\[/\(/g;
  $_ =~ s/\]/\)/g;
}
