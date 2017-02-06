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

$TREETRANS = 0;
if ($opt_t) {
  $TREETRANS = 1;
}

sub debug {
  if ($DEBUG) {
    $msg = $_[1];
    print stderr $_[0] , " " , $msg, "\n";
  }
}

## for each tree...
while ( <> ) {

  # this is not really the place for it, but should be done somewhere
  s/ +/ /g;

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

    ## tag epda begin and initial end: first -o on an eta.1 without -o in eta.1.0
    s/\^(?![^ ]*-[grie])([^ ]*) <([^>]*)> <([^ ]*-[ri][^ ]*) (?![^ ]*-[ri])([^>]*)>\^/\^\1 <\2> <\3-eb-ee \4>\^/;
    ## tag epda begin and initial end: first -g on an eta.1
    s/\^(?![^ ]*-[ge])([^ ]*) <([^>]*)> <([^ ]*-g[^ ]*) ([^>]*)>\^/\^\1 <\2> <\3-eb-ee \4>\^/;

    ## propagate epda end to eta.1 if eta has -g or -o
    s/\^([^ ]*-[gri][^ ]*)-ee([^ ]*) <([^>]*)> <([^ ]*) ([^>]*)>\^/\^\1-ei\2 <\3> <\4-ee \5>\^/;
#    ## propagate epda end to eta.1 if eta has -g but eta.1 has no -g
#    s/{([^ ]*-[go][^ ]*)-ee([^ ]*) <([^>]*)> <(?![^ ]*-g)([^ ]*) ([^>]*)>}/{\1\2 <\3> <\4-ee \5>}/;

    ####################
    ## mark current as external...
    s/\^(.*?)\^/\(\1\)/;
    ## mark first unexpanded child as current...
    s/<(.*?)>/\^\1\^/;
  }

  if ($TREETRANS) {
    ## translate to parens...
    s/\[/\(/g;
    s/\]/\)/g;
    ## for each constituent...
    while ( $_ =~ /\([^\(\)]*\)/ ) {
      ## convert outer parens to braces...
      $_ =~ s/\(([^\(\)]*)\)/\^\1\^/;
      #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
      debug($step++, "   $_");

      ## fork at -ee if more than one edge from -eb
      s/\^(?![^ ]*-eb)([^ ]*) <([^>]*)> <(?![^ ]*-eb)([^ ]*)-ee([^ ]*) ([^>]*)>\^/\^\1 <\2> <\\\3\4 *FOOT*>\^ <\3\4 \5>/;
#      ## fork at -ee: if non-escaping/terminal
#      s/{([^ ]*) <([^>]*)> <(?![^ ]*-eb)([^ ]*)-ee([^ ]*) (?!\[[^ ]*-g)([^>]*)>}/{\1 <\2> <\\\3\4 *FOOT*>} <\3\4 \5>/;
#      ## fork at -ee: if escaping
#      s/{(?![^ ]*-eb)([^ ]*)-ee([^ ]*) <([^ ]*-g[^>]*)> <([^ ]*) ([^>]*)>}/{\1\2 <\3> <\\\4 *FOOT*>} <\4 \5>/;

      ## end at middle child -eb
      s/\^([^ ]*) <([^>]*)> <([^ ]*)-eb([^ ]*) ([^>]*)> <([^>]*)>\^/\^\1 <\2> <\3\4 [\3\4 \5] [\6]>\^/;

      ## propagate triples:
      s/\^([^ ]*) <([^>]*)> <(?![^ ]*-eb)([^>]*)> <([^>]*)>\^\)/\^\1 <\2> <\3>\^ <\4>\)/;

      ####################
      ## convert inner angles (if any) to bracks...
      while ( s/(\^[^\^]*)<([^<>]*)>([^\^]*\^)/\1\[\2\]\3/ ){}
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

  ## output...
  print $_;
}
