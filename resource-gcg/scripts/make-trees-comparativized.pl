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
  return 0
}


## for each tree...
while ( <> ) {

  $line++;  #if ($line % 1000 == 0) { print stderr "$line lines processed...\n"; }
  
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

    s/ +/ /g;

    $j = 0;
    if (

#        #### expletive there...
#        (s/\^(.*?) <NP[^ ]* +\[EX +([Tt][Hh][Ee][Rr][Ee])\] *> +(.*?<VP[^>]*?(?: be| being| been| is| was|VBZ +'s| are| were| 're)\][^>]*? \[NP)(?:-PRD)?([^>]*>)(.*)\^/\^\1 <ADJP-393 \[JJ \2\]> \3-SBJ\4 <ADJP \[-NONE- \*T\*-393\]>\5\^/ && ($j=1)) ||

        #### comparatives...
        (s/\^([^ ]*) (.*)\[(RB) (as)\](.*) <(PP)( \[IN as\] [^>]*)>\^/\^\1 \2\[\3 \[\3 \4\] \[\6 \[-NONE- \*ICH\*-383\]\]\]\5 <\6-383 \7>\^/ && ($j=2)) ||
        (s/\^([^ ]*) (.*)\[(JJ) (same)\](.*) <(PP)( \[IN as\] [^>]*)>\^/\^\1 \2\[\3 \[\3 \4\] \[\6 \[-NONE- \*ICH\*-383\]\]\]\5 <\6-383 \7>\^/ && ($j=3)) ||
        (s/\^([^ ]*) (.*)\[(JJ) (different)\](.*) <(PP)( \[IN than\] [^>]*)>\^/\^\1 \2\[\3 \[\3 \4\] \[\6 \[-NONE- \*ICH\*-383\]\]\]\5 <\6-383 \7>\^/ && ($j=4)) ||
        (s/\^([^ ]*) (.*)\[(..R) ([^ ]*)\]([^>]*)( \[[^ ]* \[-NONE \*ICH\*(-[0-9]+)\]\])(.* <PP\7 [^>]*>)\^/\^\1 \2\[\3 \[\3 \4\]\6\]\5\8\^/ && ($j=5)) ||
        (s/\^([^ ]*) (.*)\[(..R) ([^ ]*)\](.*) <(PP)( \[IN than\] [^>]*)>\^/\^\1 \2\[\3 \[\3 \4\] \[\6 \[-NONE- \*ICH\*-383\]\]\]\5 <\6-383 \7>\^/ && ($j=6)) ||

        #### hyphenations...
        (s/\^([^ ]*) (\w+)\-(\w+)\^/\^\1 <\1 \2> <HYPH \-> <\1 \3>\^/ && ($j=7)) ||

#         #### Internal noun structure...
#         # Distribute '[Tt]he' preceding CC (unless followed by another '[Tt]he')...
#         (s/\^(NP[^ ]*) (<DT [^>]>|<[^ ]* [Tt][Hh][Ee]>) (.* <CC [^>]*> (?![<]DT[^>]*>|<[^ ]* [Tt][Hh][Ee]>).*)\^/\^\1 \2 <NN \3>\^/ && ($j=1)) ||
#         # Distribute 'Co.|Corp.|LLC|...' preceding CC (unless followed by another '[Tt]he')...
#         (s/\^(N[PN][^ ]*) (.* <CC [^>]*> .*) (<NNP (?:Co\.|Corp\.|Inc\.|LLC)>)\^/\^\1 <NN \2> \3\^/ && ($j=2)) ||
#         # group NNP conjs
#         (s/\^(N[PN].*?) ((?:<NNP[^>]*> *)+) (<CC[^>]*>) ((?: *<NNP[^>]*>)+)(.*)\^/\(\1 \^NNP <NNP \2> \3 <NNP \4>\^\5\)/ && ($j=4)) ||
#         # group NNP part of NP
#         (s/\^(N[PN].*?)((?: <(?:NNP|CC)[^ ]* [^ ]*>){2,}) (<(?!NNP|CC).*?)\^/\^\1 <NNP\2> \3\^/ && ($j=5)) ||
#         # group single NN|JJ conjs
#         (s/\^(N[PN].*?) ((?:<(?:NN|JJ)[^>]*> <, ,> )*<(?:NN|JJ)[^>]*> <CC[^>]*> <(?:NN|JJ)[^>]*>) (.*)\^/\^\1 <JJ \2> \3\^/ && ($j=3)) ||
# #        s/\^(N[PN].*?)((?: <(?:NNP|CC)[^ ]* [^ ]*>){2,}) (<(?!CC)[^ ]* [a-z][^ ]*>.*?)\^/\^\1 <NNP\2> \3\^/ && ($j=3)) ||
#         # group NN|JJ conjs
# #        (s/\^(N[PN].*?) ((?:<(?:NN |JJ )[^>]*> ){2}<CC[^>]*>(?: <(?:NN |JJ )[^>]*>){2}) (.*)\^/\^\1<JJ \2> \3\^/ && ($j=4)) ||

        1 ) {
        	debug($step, " used rule $j")
        }

    ####################
    while (
           #### uh-oh, turn each minimal <...> pair within other <...> into [...]
           s/(<[^>]*)<([^<>]*)>/\1\[\2\]/ ||
           0 ) {}
    ## mark current as external...
    s/\^(.*?)\^/\(\1\)/;
    ## mark first unexpanded child as current...
    s/<(.*?)>/\^\1\^/;
  }
  
  # output
  print $_;
}

