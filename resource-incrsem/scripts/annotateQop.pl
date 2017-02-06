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

getopts("dt");

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

  ## fix spurious reserved characters in linetrees file...
  s/\^/!carat!/g;
  s/</!lessthan!/g;
  s/>/!morethan!/g;
  s/\[/!openbrack!/g;
  s/\]/!closebrack!/g;
  s/[ 　]\)/!space!\)/g;

  ## mark top-level constituent as current...
  s/^ *\((.*)\) *$/\^\1\^/;
  ## mark all other constituents as internal...
  s/\(/\[/g;
  s/\)/\]/g;
  ## for each constituent...
  while ( $_ =~ /\^/ ) {
    ## mark all children of current...
    for ( $i=index($_,'^'),$d=0; $i<rindex($_,'^'); $i++ ) {
      if ( substr($_,$i,1)eq'[' ) { if($d==0){substr($_,$i,1)='<';} $d++; }
      if ( substr($_,$i,1)eq']' ) { $d--; if($d==0){substr($_,$i,1)='>';} }
    }

    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug(++$step, "   $_");

    ## convert N-b(N-aD) and N-aD-lA to N-b(N-aD)-lQ and N-aD w no -l...
    s/\^([^ ]*) +<(N-b\{N-aD\}) ([^>]*)> +<([^ ]*)-lA ([^>]*)>\^/\^\1 <\2-lQ \3> <\4 \5>\^/;
    ## convert sibling of -v... to -lV...
    s/\^(A-aN[^ ]*) +<([^ ]*) ([^>]*)> +<([^ ]*-v[^ ]*) ([^>]*)>\^/\^\1 <\2-lV \3> <\4 \5>\^/;
    s/(-xX[^ ]*)(-l[^ ]*)/\2\1/;

    ####################
    ## mark current as external...
    s/\^(.*?)\^/\(\1\)/;
    ## mark first unexpanded child as current...
    s/<(.*?)>/\^\1\^/;
  }

  ## output...
  print $_;
}
