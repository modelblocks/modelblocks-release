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


# cat wsj_0001.trees | perl scripts/treesed.pl
use Getopt::Std;

getopts("d");

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
  ## translate to parens...
  s/\[/\(/g;
  s/\]/\)/g;
  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    $_ =~ s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug($step++, "   $_");
    #### find modifier embeddings
    s/{([^ ]*-d1[^ ]* <[^ ]*-d2[^ ]* [^>]*> <[^ ]*-lM[^ ]* [^>]*>)}/{FOUNDMOD \1}/g;
    s/{([^ ]*-d2[^ ]* <[^ ]*-d3[^ ]* [^>]*> <[^ ]*-lM[^ ]* [^>]*>)}/{FOUNDMOD \1}/g;
    s/{([^ ]*-d3[^ ]* <[^ ]*-d4[^ ]* [^>]*> <[^ ]*-lM[^ ]* [^>]*>)}/{FOUNDMOD \1}/g;

    #### find conj embeddings
    s/{([^ ]*-d1[^ ]* <[^ ]*-lC[^ ]*-d2[^ ]* [^>]*> <[^ ]* [^>]*>)}/{FOUNDCONJ \1}/g;
    s/{([^ ]*-d2[^ ]* <[^ ]*-lC[^ ]*-d3[^ ]* [^>]*> <[^ ]* [^>]*>)}/{FOUNDCONJ \1}/g;
    s/{([^ ]*-d3[^ ]* <[^ ]*-lC[^ ]*-d4[^ ]* [^>]*> <[^ ]* [^>]*>)}/{FOUNDCONJ \1}/g;

    #### find canonical embeddings
    s/{([^ ]*-d1[^ ]* <(?![^ ]*-lC)[^ ]*-d2[^ ]* \[[^>]*> <(?![^ ]*-lM[^ ]*)[^>]*>)}/{FOUNDCANON \1}/g;
    s/{([^ ]*-d2[^ ]* <(?![^ ]*-lC)[^ ]*-d3[^ ]* \[[^>]*> <(?![^ ]*-lM[^ ]*)[^>]*>)}/{FOUNDCANON \1}/g;
    s/{([^ ]*-d3[^ ]* <(?![^ ]*-lC)[^ ]*-d4[^ ]* \[[^>]*> <(?![^ ]*-lM[^ ]*)[^>]*>)}/{FOUNDCANON \1}/g;

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/({[^{}]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    $_ =~ s/{(.*)}/<\1>/;
  }

  #################### FINALLY: delete numbers...
  s/[-=][0-9]+([^ \)\]]*) /\1 /g;
  s/[-=][0-9]+([^ \)\]]*) /\1 /g;
  s/[-=][0-9]+([^ \)\]]*) /\1 /g;

  ## finish up...
  $_ =~ s/</[/;
  $_ =~ s/>/]/;
  ## translate to parens again...
  $_ =~ s/\[/\(/g;
  $_ =~ s/\]/\)/g;

  #### seriously: I don't want any empty categories!
  s/ \([^ ]* +\([^ ]* +\([^ ]* \)\)\)//g;
  s/ \([^ ]* +\([^ ]* \)\)//g;
  s/ \([^ ]* \)//g;

  ## output...
  print $_;
}
