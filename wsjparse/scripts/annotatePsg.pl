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



#### NOTE!
#s/-[mnp]Q//g;

## for each tree...
while ( <> ) {

  $line++;  if ($line % 1000 == 0) { print stderr "$line lines processed...\n"; }

  #### category normalization
  # root category
  s/^\((?!NP|FRAG)/\(SS-f/;
  s/^\(FRAG/\(AP-fFRAG/;
  s/^\(NP(?!-f)/\(NP-fNP/;
  # use CONJP for "as well as"
  s/(\([^ ]* as\) \([^ ]* well\) \([^ ]* as\))/\(CONJP \1\)/g;
#  # use CONJP for "yet"
#  s/\([^ ]* (\([^ ]* yet\))\)/\(CONJP \1\)/g;
  # use CC for CONJP
  s/\(CONJP/\(CC/g;
#  s/\((NNPS|NNP|NNS|NN)/\(NN/g;


  $INIT_PUNCT = '-LRB-|-LCB-|``|`|--|-|,|;|\.|!|\?|\.\.\.|\'|\'\'|-RCB-|-RRB-';
  $FINAL_PUNCT = '-LRB-|-LCB-|``|`|--|-|,|;|:|\.|!|\?|\.\.\.|\'|\'\'|-RCB-|-RRB-';
  $LP_FIRST = 'got|made|Been|been|led|become|taken|learned|attacked|given|won|waffled|had|changed|put\]|hurt\]|settled|tried|managed|worked|grown\] *\[PP|done\] *\[NP *\[NP|written\] *\[NP *\[NP';
  $NONVPS_VP_NONVBS = '(?:(?=\[VP)[^>])*\[VP(?:(?=\[VB)[^>])*\[';
  $EMPTY_SUBJ = '\[NP[^ ]* \[-NONE- \*[^A-Z ]*\]\]';
  $NO_VP_HEAD = '(?:(?!\[VB|\[JJ|\[MD|\[TO)[^>])*';
  $NO_AP_HEAD = '(?:(?!\[VB|\[JJ|\[IN|\[MD|\[TO)[^>])*';
  $SUITABLE_REL_CLAUSE = '[^\]]*\[WH[^ ]*-[0-9]+ \[(?!-NONE-)[^ ]* (?!what)[^>]*';

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    #debug($step++, "   $_");

    # turn conj of null-subj-S's with null subj into null-subj-S with conj of VP's
    s/{S([^ ]*) <S(?:[-=][^ ]*)? (\[NP[^ ]* \[-NONE- [^\]]*\]\]) ([^>]*)> (.*<CC[^>]*>.*) <S(?:[-=][^ ]*)? \2 ([^>]*)>}/{S\1 \2 <VP \3 \4 \5>}/;
    # turn mod of null-subj-S into null-subj-S with mod of VP
    s/{S(?![A-Z])([^ ]*) (<.*) (<NP[^ ]* \[-NONE- [^\]]*\]>) (<VP.*)}/{S\1 \3 \2 \4}/;

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/({[^{}]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    s/{(.*)}/<\1>/;
  }
  ## finish up...
  s/</[/;
  s/>/]/;
  ## translate to parens again...
  s/\[/\(/g;
  s/\]/\)/g;


  ######################################################################
  ## I. BOTTOM-UP: PROPAGATE LEX CAT UP TO S NODE

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...

    # propagate V up to VP node to determine VP/IP/BP/LP/AP...
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<(?:VB[ZDP]|MD).*)}/{VP-TOBEVP\1 \2}/;
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<TO.*)}/{VP-TOBEIP\1 \2}/;
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<VB .*)}/{VP-TOBEBP\1 \2}/;
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<VB[GN].*)}/{VP-TOBEAP\1 \2}/;
    # propagate TOBE.P VP up through VP conj...
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*<VP[^ ]*-TOBE([VIBLA])P.*<CC.*)}/{VP-TOBE\3P\1 \2}/;
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*<CC.*<VP[^ ]*-TOBE([VIBLA])P.*)}/{VP-TOBE\3P\1 \2}/;
    # propagate first TOBE.P VP up through VP mod...
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*?<VP[^ ]*-TOBE([VIBLA])P.*)}/{VP-TOBE\3P\1 \2}/;
    # propagate empty VP as TOBEVP
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? (<-NONE-[^>]*>)}/{VP-TOBEVP\1 \2}/;
    # propagate random VP as TOBEAP
    s/{VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?!.*<VP.*\}).*)}/{VP-TOBEAP\1 \2}/;
    # propagate PRD XP as TOBEAS
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VP).)*.*<[^ ]*-PRD.*)}/{S-TOBEAS\1 \2}/;
debug($step, ":( $_") ||
    # propagate TOBE.P VP with empty NP up through S...
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VP).)*<NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]> (?:(?!<VP).)*<VP[^ ]*-TOBE([VIBLA])P.*)}/{S-TOBE\3P\1 \2}/;
    #s/{S(?![^ ]*-TOBE|[^ ]*-NOM)([-=][^ ]*)? ((?:(?!<VP).)*)<NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]> ((?:(?!<VP).)*<VP[^ ]*-TOBE([VIBLA])P.*)}/{S-TOBE\4P\1 \2\3}/;
    # propagate TOBE.P VP up through S...
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VP).)*.*<VP[^ ]*-TOBE([VIBLA])P.*)}/{S-TOBE\3S\1 \2}/;
    # propagate TOBE.S S up through S conj...
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? (.*<S[^ ]*-TOBE(..).*<CC.*)}/{S-TOBE\3\1 \2}/;
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? (.*<CC.*<S[^ ]*-TOBE(..).*)}/{S-TOBE\3\1 \2}/;
    # propagate first TOBE.S S up through S mod...
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? (.*?<S[^ ]*-TOBE(..).*)}/{S-TOBE\3\1 \2}/;
    # propagate empty S as TOBEVS
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? (<-NONE-[^>]*>)}/{S-TOBEVS\1 \2}/;
    # propagate random S as TOBEAS
    s/{S(?![^ ]*-TOBE)([-=][^ ]*)? ((?!.*<VP.*\}).*)}/{S-TOBEAS\1 \2}/;

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/({[^{}]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    s/{(.*)}/<\1>/;
  }
  ## finish up...
  s/</[/;
  s/>/]/;
  ## translate to parens again...
  s/\[/\(/g;
  s/\]/\)/g;


  ######################################################################
  ## II. TOP-DOWN: REANNOTATE TO PSG FORMAT

  ## mark top-level constituent as current...
  s/^ *\((.*)\) *$/{\1}/;
  ## mark all other constituents as internal...
  s/\(/\[/g;
  s/\)/\]/g;
  ## for each constituent...
  while ( $_ =~ /{/ ) {
    ## mark all children of current...
    for ( $i=index($_,'{'),$d=0; $i<index($_,'}'); $i++ ) {
      if ( substr($_,$i,1)eq'[' ) { if($d==0){substr($_,$i,1)='<';} $d++; }
      if ( substr($_,$i,1)eq']' ) { $d--; if($d==0){substr($_,$i,1)='>';} }
    }
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug(++$step, "   $_");


    ######################################################################
    ## A. COLLAPSE ALL UNARY CONSTITUENTS IN ORIGINAL TREEBANK, KEEPING HIGHEST CAT LABEL

#    while (
#           debug($step, "-- $_") ||
#           #### collapse unary constituents using upper cat, lower -f
#           s/\(([^ ]*-f)[^ ]* \{(?![^ ]*-f)(?![^>\]]*-NONE-[^<\[]*\})(.*)\}\)/{\1\2}/ ||
#           0 ) {}
    while (
           debug($step, "-- $_") ||
           #### collapse unary constituents using upper cat, lower -f
           s/{([^ ]*-f)[^ ]* <(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>}/<\1\2>/ ||
           0 ) {}


    ######################################################################
    ## B. BRANCH OFF MODIFIERS AND ARGUMENTS IN ORDER OF PRECEDENCE

    #### only one rewrite rule may apply at any node (recurse only if new node created)
    #### rewrite rules are ordered by precedence: rules preferred at higher nodes are first
    if (
        debug($step, ".. $_") ||

#        #### relabel 'S' constituents by making them 'transparent' so sub-constituents can be identified
#        # classify transparent constituents
##        s/{(.*) \|S([^ ]*-NOM[^\|]*)\|(.*)}/<\1 <SheadN\2>\3>/ ||
###        s/{(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<VP$NO_VP_HEAD \[TO [^\|]*)\|(.*)}/<\1 <SheadInosubj\2>\3>/ ||
###        s/{(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<(?:VP|ADJP)$NO_VP_HEAD \[(?:VB[NG]|JJ[RS]*) [^\|]*)\|(.*)}/<\1 <SheadAnosubj\2>\3>/ ||
###        s/{(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<[^ ]*-PRD[^\|]*)\|(.*)}/<\1 <SheadAnosubj\2>\3>/ ||
#        s/{(.*) \|S([^\|]*<VP$NO_VP_HEAD \[VB [^\|]*)\|(.*)}/<\1 <SheadB\2>\3>/ ||
#        s/{(.*) \|S([^\|]*<VP$NO_VP_HEAD \[TO [^\|]*)\|(.*)}/<\1 <SheadI\2>\3>/ ||
#        s/{(.*) \|S([^\|]*<(?:VP|ADJP)$NO_VP_HEAD \[(?:VB[NG]|JJ[RS]*) [^\|]*)\|(.*)}/<\1 <SheadA\2>\3>/ ||
#        s/{(.*) \|S([^\|]*<[^ ]*-PRD[^\|]*)\|(.*)}/<\1 <SheadA\2>\3>/ ||
#        s/{(.*) \|S([^\|]*)\|(.*)}/<\1 <SheadV\2>\3>/ ||
#        # create transparent constituents
#        s/{(.*) <S(?!-NOM)(?!head)(?![A-Z])([^>]*)>(.*)}/<\1 \|S\2\|\3>/ ||

        ######################################################################
        ## 1. UNARY REANNOTATIONS

        #### remove empty NP from S turned to [VIBLA]P
        s/{([VIBLAR]P[^ ]*.*) <NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]>(.*)}/<\1\2>/ ||

        #### parentheticals
        # identify ptb tag dominating its own trace, and change it to *INTERNAL*
        s/{([^ ]*)(-f[^ ]*)(-[0-9]+)((?![0-9]).*-NONE- \*)(?:T|ICH)(\*\3(?![0-9]).*)}/<\1\2\3\4INTERNAL\5>/ ||
        # flatten PRN nodes
        s/{(.*) <PRN([^>]*)>(.*)}/<\1\2\3>/ ||

        ######################################################################
        ## 2. HIGH PRECEDENCE BRANCHING FINAL CONSTITUENT

        #### SS
        # semi/dash splice between matching constituents
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) <([^- ]*)([^>]*)> <([^ ]*) (;|--)> <\4([^>]*)>}/{\1\2\3 <\1\2-lC-f\4\5> <Cs\1-lI-fNIL <\6-lI-f\6 \7> <\1\2-lC-f\4\8>>}/ ||
#        # inverted sentence: branch off final raised complement SS (possibly quoted) with colon
#        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- *\*ICH\*(-[0-9]+)(?![0-9]).*) <: :> <(S)([^ ]*)\5([^>]*)>}/{\1\2\3 <VS-hSS\5\2-lI-fNIL \4> <SSmod-lI-f\6\7 <: :> <SS-lI-f\6\7\5\8>>}/ ||
        # branch off final SBAR as extraposed modifier AC rel clause:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gAC ...> <AC WH# ... t# ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 $SUITABLE_REL_CLAUSE)>}/{\1\2\3 <\1\2-gAC\5-lI-fNIL \4> <AC-oR-lN-f\6>}/ ||
        # branch off final SBAR as extraposed modifier IP:                           {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gRP ...> <IP WH# ... t# ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*) \[ADVP \[-NONE- \*T\*\7\]\]([^>]*)\]>}/{\1\2\3 <\1\2-gAC\5-lI-fNIL \4> <IP-lN-f\6 \8\9>}/ ||
        # branch off final SBAR as extraposed modifier VE complement:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gAC ...> <VE ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 \[IN that\][^>]*)>}/{\1\2\3 <\1\2-gVE\5-lI-fNIL \4> <VE-lN-f\6>}/ ||
        # inverted sentence: branch off final raised complement SS (possibly quoted)
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(S)([^ ]*)\5([^>]*)>}/{\1\2\3-modeverused? <VS-gSS\5\2-lI-fNIL \4> <SS-lN-f\6\7\5\8>}/ ||

        # branch off final punctuation (passing -o to head)
        s/{(SS|EQ|VQ|VE|IE|AC|RC|[VIBLAGR][SP]|NP|AA|RR|NN)([^ ]*?)(-[fghilp][^ ]*) (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>}/ ||

        # branch off final possessive 's
        s/{(DD|NP)([^ ]*) (<.*) <(POS 's?)>}/{\1\2 <NP-lA-fNIL \3> <DD-sNP-lI-f\4>}/ ||

        ######################################################################
        ## 3. HIGH PRECEDENCE BRANCHING INITIAL CONSTITUENTS

        # branch off initial punctuation (passing -o to head)
        s/{(SS|EQ|VQ|VE|IE|AC|RC|[VIBLAGR][SP]|NP|AA|RR|NN)([^ ]*?)(-[fghilp][^ ]*) <([^ ]*) ($INIT_PUNCT)> (<.*)}/{\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>}/ ||

        #### AP|RP
        # unary expand to NP
        s/{(AP|RP)([^ ]*?)(-[g][^ ]*)?(-fNP[^ ]*) (<.*)}/{\1\2\3\4 <NP\2-lI-fNIL \5>}/ ||
        # unary expand to NP (nom clause)
        s/{(AP)([^ ]*?)(-[g][^ ]*)?(-fSBAR[^ ]*(?= <WH)) (<.*)}/{\1\2\3\4 <NP\2-lI-fNIL \5>}/ ||
        # branch off initial specifier NP measure
        s/{(AP|RP)([^ ]*?)(-[fghilop][^ ]*) <(NP[^>]*)> (<.*)}/{\1\2\3 <NP-lA-f\4> <\1\2-sNP-lI-fNIL \5>}/ ||
        # delete initial empty subject
        s/{(RP)([^ ]*?)(-[fghilop][^ ]*) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)}/<\1\2\3 \4>/ ||
#        # branch off initial modifier RR
#        s/{(RP)([^ ]*?)(-[fghilop][^ ]*) <(RB|ADVP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3 <RR-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
#        s/{(AP|RP)([^ ]*?)(-[fghilop][^ ]*) <(RB|ADVP|ADJP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO).*)}/{\1\2\3 <RR-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
##           s/{(AP|RP)([^ ]*)(-f[^ ]*) (<TO.*<VP.*)}/{\1\2\3 <IP\2-fNIL \4>}/ ||

        #### initial filler
        # content question: branch off initial interrogative NP
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NP-oI-lN-f\4\5\6\7> <VQ\2-gNP\6-lI-fNIL \8>}/ ||
        # content question: branch off initial interrogative RP
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-oI-lN-f\4\5\6\7> <VQ\2-gRP\6-lI-fNIL \8>}/ ||
        # topicalized sentence: branch off initial topic SS (possibly quoted)
        s/{(SS|VE|IE|VS)([^ ]*?)(-[fghilop][^ ]*) (?!<[^ ]*-SBJ)<(S)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <SS-lN-f\4\5\6\7> <VS\2-gSS\6-lI-fNIL \8>}/ ||
        # topicalized sentence: branch off initial topic NP   ***<[^ ]* \[-NONE- [^\]]*\]>|
        s/{(SS|VE|IE|VS)([^ ]*?)(-[fghilop][^ ]*) (?!<[^ ]*-SBJ)<(NP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NP-lN-f\4\5\6\7> <VS\2-gNP\6-lI-fNIL \8>}/ ||
        # topicalized sentence: branch off initial topic AP
        s/{(SS|VE|IE|VS)([^ ]*?)(-[fghilop][^ ]*) (?!<[^ ]*-SBJ)<(ADJP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <AP-lN-f\4\5\6\7> <VS\2-gAP\6-lI-fNIL \8>}/ ||
        # topicalized sentence: branch off initial topic RP
        s/{(SS|VE|IE|VS)([^ ]*?)(-[fghilop][^ ]*) (?!<[^ ]*-SBJ)<((?!WH)[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-lN-f\4\5\6\7> <VS\2-gRP\6-lI-fNIL \8>}/ ||
        # embedded sentence: delete initial empty complementizer
        s/{(VE)([^ ]*?)(-[fghilop][^ ]*) <-NONE-[^>]*> (<.*)}/{\1\2\3 <VS\2-lI-fNIL \4>}/ ||
        # embedded sentence: branch off initial complementizer
        s/{(V|I)E([^ ]*?)(-[fghilop][^ ]*) <(IN[^>]*)> (<.*)}/{\1E\2\3 <\1E\2-u\1S-lM-f\4> <\1S-lI-fNIL \5>}/ ||
        # embedded noun: branch off initial preposition
        s/{(N)E([^ ]*?)(-o[A-Z])?(-[fghilp][^ ]*) <(IN[^>]*)> (<.*)}/{\1E\2\3\4 <\1E\2-u\1P-lM-f\5> <\1P\3-lI-fNIL \6>}/ ||
        # embedded question: branch off initial interrogative RP whether/if
        s/{(EQ)([^ ]*?)(-[fghilop][^ ]*) <(IN[^>]*)> (<.*)}/{\1\2\3 <RP-oI-lI-f\4> <VS\2-gRP-lN-fNIL \5>}/ ||

        #### initial RP/RC modifier
        # branch off initial modifier RP with colon
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S)([^ ]*?)(-[fghilop][^ ]*) <(PP|RB|ADVP|CC|FRAG|NP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> <([^ ]*) (:)> (<.*)}/{\1\2\3 <RP-lM-fNIL <RP-lI-f\4\5> <\6 \7>> <\1\2-lI-fNIL \8>}/ ||
        # branch off initial modifier RP IP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghilop][^ ]*) <S[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ($NO_AP_HEAD\[TO[^>]*)> (<.*)}/{\1\2\3 <RP-lM-fNIL <IP-lI-fNIL \4>> <\1\2-lI-fNIL \5>}/ ||
        # branch off initial modifier RP AP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghilop][^ ]*) <S[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ([^>]*)> (<.*)}/{\1\2\3 <RP-lM-fNIL \4> <\1\2-lI-fNIL \5>}/ ||
        # branch off initial modifier RP from SBAR
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghilop][^ ]*) <(SBAR(?![^ ]*-SBJ)[^ ]* (?!\[IN that|\[IN for|\[IN where|\[IN when)(?!\[WH[^  ]*))([^>]*)> (<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        # branch off initial modifier RC from SBAR-ADV
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghilop][^ ]*) <(SBAR(?:-ADV|-TMP)[^ ]* \[WH)([^>]*)> (<.*)}/{\1\2\3 <RC-oR-lN-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        # branch off initial RB + JJS as modifier RP  (e.g. "at least/most/strongest/weakest")
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P|NP)([^ ]*?)(-[fghilop][^ ]*) (<IN[^>]*> <JJ[^>]*>) (?!<CC)(<.*)}/{\1\2\3 <RP-lM-fNIL \4> <\1\2-lI-fNIL \5>}/ ||
        # branch off initial modifier RP  (incl determiner, e.g. "both in A and B")
#WORSE:        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)(-[fghilop][^ ]*) <([^ ]*-TMP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3 <RP-t-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghilop][^ ]*) <(DT|PP|RB|IN|ADVP|CC|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        s/{(NP)([^ ]*?)(-o[A-Z])?([^ ]*?)(-[fghilop][^ ]*) <(CC)([^>]*)> (<(?!PP|WHPP).*)}/{\1\2\3\4\5 <RR-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ ||
        s/{(NP)([^ ]*?)(-o[A-Z])?([^ ]*?)(-[fghilop][^ ]*) <(RB|PDT|(?:CC|DT) (?:[Nn]?[Ee]ither|[Bb]oth)(?=.*<CC.*\})|DT (?:[Aa]ll|[Bb]oth|[Hh]alf)|(?:ADJP|QP)(?=[^>]*\[DT [^\]]*\]>))([^>]*)> (<.*)}/{\1\2\3\4\5 <RR-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ ||
        s/{(NP)([^ ]*?)(-o[A-Z])?([^ ]*?)(-[fghilop][^ ]*) <(ADVP|PP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3\4\5 <RP-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ ||
#        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)(-[fghilop][^ ]*) <(S)([^ ]* \[NP[^ ]* \[-NONE- \*-[^>]*)> (<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        # branch off initial modifier RR-oI/R of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        s/{(AP|RP)([^ ]*?)(-o[A-Z])(-[fghilp][^ ]*) <(WRB)([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3\4 <RR\3-lM-f\5\6> <\1\2-lI-fNIL \7>}/ ||
        # branch off initial modifier RR of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        s/{(AP|RP)([^ ]*?)(-[fghilp][^ ]*) <(DT|PP|RB|IN(?=[^\|]*<IN)|ADVP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3 <RR-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||

        #### CODE REVIEW: WHADVP/WP$ in NP needs to inherit -o  ******************

        #### sentence types
        # branch off initial parenthetical sentence with extraction
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)}/{\1\2\3 <VS-gSS\6-lN-f\4\5> <\1\2-lI-fNIL \7>}/ ||
        # branch off initial parenthetical sentence w/o extraction
        s/{(VP)([^ ]*?)(-[fghilop][^ ]*) <(S(?![A-Z]))([^>]*)> (.*<VP.*)}/{\1\2\3 <VS-lN-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        # imperative sentence: delete empty NP
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) <NP[^ ]* \[-NONE- \*\]> (<VP$NO_VP_HEAD \[VB.*)}/{\1\2\3 <BP\2-lI-fNIL \4>}/ ||
        # declarative (inverted or uninverted) sentence: unary expand to VS
        s/{(SS|VE)([^ ]*?)(-[fghilop][^ ]*) (<(?:NP|[^ ]*-SBJ).*<VP.*|<VP.*<(?:NP|[^ ]*-SBJ))}/{\1\2\3 <VS\2-lI-fNIL \4>\5}/ ||
        # polar question: unary expand to VQ
        s/{(SS|EQ)([^ ]*?)(-[fghilop][^ ]*) (<[^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Hh]as|[Hh]ave|[Hh]ad)).*<NP.*)}/{\1\2\3 <VQ\2-lI-fNIL \4>}/ ||
        # imperative sentence: unary expand to BP   ***PROBABLY NULL CAT HERE***
        s/{(SS)([^ ]*?)(-[fghilop][^ ]*) (<VP$NO_VP_HEAD \[VB.*)}/{\1\2\3 <BP\2-lI-fNIL \4>}/ ||
        # embedded question / nom clause: branch off initial interrogative NP and final modifier IP with NP gap (what_i to find a picture of t_i)
        s/{(EQ|NP)([^ ]*?)(-[fghilop][^ ]*) <(WHNP[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>}/{\1\2\3 <NP-oI-lI-f\4\5\6> <IP-gNP\5-lN-fNIL \7>}/ ||
        # embedded question / nom clause: branch off initial interrogative RP and final modifier IP with RP gap (how_i to find a picture t_i)
        s/{(EQ|NP)([^ ]*?)(-[fghilop][^ ]*) <(WH[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>}/{\1\2\3 <RP-oI-lI-f\4\5\6> <IP-gRP\5-lN-fNIL \7>}/ ||
        # embedded question / nom clause: branch off initial interrogative NP
        s/{(EQ|NP)([^ ]*?)(-[fghilop][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NP-oI-lI-f\4\5\6\7> <VS\2-gNP\6-lN-fNIL \8>}/ ||
        # embedded question / nom clause / nom clause modifier: branch off initial interrogative RP
        s/{(EQ|NP)([^ ]*?)(-[fghilop][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-oI-lI-f\4\5\6\7> <VS\2-gRP\6-lN-fNIL \8>}/ ||
        # polar question: branch off initial BP-taking auxiliary
        s/{(VQ)([^ ]*?)(-[fghilop][^ ]*) <([^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|'d))[^>]*)> (<.*)}/{\1\2\3 <\1\2-uBS-lM-f\4> <BS-lI-fNIL \5>}/ ||
        # polar question: branch off initial NP-taking auxiliary
        s/{(VQ)([^ ]*?)(-[fghilop][^ ]*) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<(?:NP|NN|DT)[^>]*>)}/{\1\2\3 <\1\2-uNP-lM-f\4> <NP-lI-fNIL \5>}/ ||
        # polar question: branch off initial AP-taking auxiliary
        s/{(VQ)([^ ]*?)(-[fghilop][^ ]*) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<.*)}/{\1\2\3 <\1\2-uAS-lM-f\4> <AS-lI-fNIL \5>}/ ||
        # polar question: branch off initial LP-taking auxiliary  ***NOTE: 's AND 'd WON'T GET USED***
        s/{(VQ)([^ ]*?)(-[fghilop][^ ]*) <([^\]]*VB[A-Z]? (?:[Hh]as|[Hh]ave|[Hh]ad|'s|'ve|'d)[^>]*)> (<.*)}/{\1\2\3 <\1\2-uLS-lM-f\4> <LS-lI-fNIL \5>}/ ||
        # polar question: allow subject gap without inversion
        s/{(VQ)([^ ]*-gNP(-[0-9]+)[^ ]*?)(-[fghilop][^ ]*) <NP[^ ]* \[-NONE- \*T\*\3\]> (<.*)}/{\1\2\4 <VP-fNIL \5>}/ ||
        # embedded sentence: delete initial empty interrogative phrase     ****WHY WOULD THIS HAPPEN??? NO WH IN EMBEDDED SENTENCE****
        s/{(VE)([^ ]*?)(-[fghilop][^ ]*) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gNP\4-lI-fNIL \5>}/ ||

        #### rel clause
        # implicit-pronoun relative: delete initial empty interrogative phrase
        s/{(AC|RC)([^ ]*?)(-[fghilop][^ ]*) <WHNP[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gNP\4-lI-fNIL \5>}/ ||
        # implicit-pronoun relative: delete initial empty interrogative phrase as adverbial
        s/{(AC|RC)([^ ]*?)(-[fghilop][^ ]*) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gRP\4-lI-fNIL \5>}/ ||
        # branch off initial relative noun phrase
        s/{(AC|RC|RP)([^ ]*?)(?:-oR)?(-[fghilop][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2-oR\3 <NP-oR-lN-f\4\5\6\7> <VS\2-gNP\6-lI-fNIL \8>}/ ||
        # branch off initial relative adverbial phrase with empty subject ('when in rome')
        s/{(AC|RC|RP)([^ ]*?)(?:-oR)?(-[fghilop][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (.*\[-NONE- *\*T\*\6\].*)>}/{\1\2-oR\3 <RP-oR-lN-f\4\5\6\7> <AP\2-gRP\6-lI-fNIL \8>}/ ||
        # branch off initial relative adverbial phrase
        s/{(AC|RC|RP)([^ ]*?)(?:-oR)?(-[fghilop][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2-oR\3 <RP-oR-lN-f\4\5\6\7> <VS\2-gRP\6-lI-fNIL \8>}/ ||
        # embedded question: branch off initial interrogative RP whether/if
        s/{(AC|RC)([^ ]*?)(?:-oR)?(-[fghilop][^ ]*) <(IN[^>]*)> (<.*)}/{\1\2-oR\3 <RP-oI-lI-f\4> <VS\2-gRP-lN-fNIL \5>}/ ||

        #### middle NP
        # branch off middle modifier AP colon
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<[^ ]* :> <.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL \5>}/ ||

        #### conjunction
        # branch final right-node-raising complement NP
        s/{()([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <(NP[^ ]*\5[^>]*)>}/{\1\2\3 <\1\2-hNP\5-lI-fNIL \4> <NP-lA-f\6>}/ ||
        # branch final right-node-raising modifier AP
        s/{()([^ ]*?)(-[fghilop][^ ]*) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <((?:PP)[^ ]*\5[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\6>}/ ||
        # pinch ... CC ... -NONE- and re-run
        s/{([^C][^ ]*?)(-[fghilop][^ ]*)(?!.*\|) (<.*) (<CC[^>]*>) (<.*) (<[^ ]* \[-NONE- [^\]]*\]>)}/<\1\2 <\1 \3 \4 \5> \6>/ ||
        # branch off initial colon in colon...semicolon...semicolon construction
        s/{([^ ]*)(-[fghilop][^ ]*)(?!.*\|) (<. :>) <([^ ]*)([^>]*)> (<. ;.*<. ;.*)}/{\1\2 \3 <\4-lA-fNIL <\4\5> \6>}/ ||
        # branch off initial conjunct prior to semicolon delimiter
        s/{([^C][^ ]*?)(-[fghilop][^ ]*)(?!.*\|) (<.*?) (<[^ ]* ;> .*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <Cs\1-lI-fNIL \4>}/ ||
        # branch off initial conjunct prior to comma delimiter
        s/{([^C][^ ]*?)(-[fghilop][^ ]*)(?!.*\|) (<.*?) (<[^ ]* ,> .*<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <Cc\1-lI-fNIL \4>}/ ||
        # branch off initial conjunct prior to conj delimiter
        s/{([^C][^ ]*?)(-[fghilop][^ ]*)(?!.*\|) (<.*?) (<CC[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <C\1-lI-fNIL \4>}/ ||
#        # branch off initial conjunct prior to comma/semi/colon/dash between matching constituents
#        s/{([^C][A-Z]S[^ ]*?)(-[fghilop][^ ]*) <([^- ]*)([^>]*)> (<[^ ]* ,> <\3[^>]*>)}/{\1mod\2 <\1-lC-f\3\4> <C\1-lI-fNIL \5>}/ ||
        # branch off initial semicolon delimiter
        s/{(Cs)([^- ]*)([^ ]*?)(-[fghilop][^ ]*) <([^ ]*) (;)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;).*)}/{\1\2\3\4 <\5-lM-f\5 \6> <\1\2-pS\3-lI-fNIL \7>}/ ||
        # branch off initial comma delimiter
        s/{(Cc)([^- ]*)([^ ]*?)(-[fghilop][^ ]*) <([^ ]*) (,)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not).*)}/{\1\2\3\4 <\5-lM-f\5 \6> <\1\2-pC\3-lI-fNIL \7>}/ ||
        # branch off initial conj delimiter and final conjunct (and don't pass -p down)
        s/{(C[sc])([^- ]*)([^ ]*?)(-p[SC])(-[fghilop][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <\2\3-lC-fNIL \7>}/ ||
        # branch off initial conj delimiter and final conjunct (no -p to remove)
        s/{(C[sc]?)([^- ]*)([^ ]*?)()(-[fghilop][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <\2\3-lC-fNIL \7>}/ ||
#        # branch off initial comma/semi/colon/dash between matching constituents
#        s/{(C)([^sc][^- ]*)([^ ]*?)()(-[fghilop][^ ]*) (<[^ ]* (?:,|;|:|--|-)>) (<.*)}/{\1mod\2\3\4\5 \6 <\2\4-lI-fNIL \7>}/ ||
        # branch off initial conjunct prior to semicolon delimiter
        s/{(Cs)([^- ]*)([^ ]*?)(-pS)(-[fghilop][^ ]*) (<.*?) (<[^ ]* ;> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ ||
        # branch off initial conjunct prior to comma delimiter
        s/{(Cc)([^- ]*)([^ ]*?)(-pC)(-[fghilop][^ ]*) (<.*?) (<[^ ]* ,> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ ||
        # branch off initial conjunct prior to conj delimiter (and don't pass -p down)
        s/{(C[sc])([^- ]*)([^ ]*?)(-p[SC])(-[fghilop][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ ||
        # branch off initial conjunct prior to conj delimiter (no -p to remove)
        s/{(C[sc]?)([^- ]*)([^ ]*?)()(-[fghilop][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ ||

        ######################################################################
        ## 4. LOW PRECEDENCE BRANCHING FINAL CONSTITUENTS

        # branch off final parenthetical sentence with extraction
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <VS-gSS\7-lN-f\5\6>}/ ||

        # branch NP -> DD AA: 'the best' construction
        s/{(NP)([^ ]*?)(-[fghilop][^ ]*) <(?:DT)([^>]*)> <(?:RB|ADJP)([^>]*)>}/{\1\2\3 <DD-lM-f\4> <AA-lI-f\5>}/ ||

        #### final VP|IP|BP|LP|AP (following auxiliary)
        # branch off final VP as argument BP
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*(?:TO |MD | do[\]>]| does[\]>]| did[\]>]).*?) (<RB.*)?(<VP.*>)}/{\1\2\3 <\1\2-uBP-lM-fNIL \4> <BP-lI-fNIL \5\6>}/ ||
        # branch off final VP as argument LP (w. special cases b/c 's ambiguous between 'has' and 'is')
        s/{([VIBLAG]P)([^ ]*?)(-[fghilop][^ ]*) (.*(?: have| having| has| had| 've|VBD 'd)>.*?) (<RB.*)?(<VP.*>)}/{\1\2\3 <\1\2-uLP-lM-fNIL \4> <LP-lI-fNIL \5\6>}/ ||
        s/{([VIBLAG]P)([^ ]*?)(-[fghilop][^ ]*) (.*<VBZ *'s>.*?) (<RB.*)?(<VP[^\]]* (?:$LP_FIRST).*>)}/{\1\2\3 <\1\2-uLP-lM-fNIL \4> <LP-lI-fNIL \5\6>}/ ||
        # branch off final PRT as argument AP particle
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(PRT)([^ ]*) \[RP ([^ ]*)\]>}/{\1\2\3 <\1\2-uAP\7-lI-fNIL \4> <AP\7-lA-f\5\6 \7>}/ ||
        # branch off final modifier RP (extraposed from argument)    **TO PRESERVE EXTRAPOSN: /{\1\2\3 <\1\2-gRP\5-lI-fNIL \4> <RP-lM-f\5\6\7>}/ ||
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<VB.*<.*\[-NONE- \*ICH\*(-[0-9]+)(?![0-9]).*) <(VP[^ ]*)\5((?![0-9])[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\6\5\7>}/ ||
        # branch off final VP|ADJP as argument AP
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (.*(?: be| being| been| is| was|VBZ 's| are| were| 're)>.*?) (<RB.*)?(<(?:VP|VB[DNG]|ADJP|JJ|CD|PP[^ ]*-PRD|IN|UCP|ADVP[^ ]*-PRD|SBAR[^ ]*-PRD (?!\[WH|\[IN that)).*>)}/{\1\2\3 <\1\2-uAP-lM-fNIL \4> <AP-lI-fNIL \5\6>}/ ||
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (.*(?: be| being| been| is| was|VBZ 's| are| were| 're)>.*?) (<RB.*)?(<(?:NP|NN|S[^ ]*-NOM|SBAR[^ ]*-NOM|SBAR[^ ]*-PRD).*>)}/{\1\2\3 <\1\2-uAP-lM-fNIL \4> <AP-lI-fNIL <NP-lI-fNIL \5\6>>}/ ||
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(ADJP|PRT|ADVP[^ ]*-PRD|PP[^ ]*-PRD|VP$NO_VP_HEAD \[VB[NG])([^>]*)>}/{\1\2\3 <\1\2-uAP-lI-fNIL \4> <AP-lA-f\5\6>}/ ||
        s/{(AP|RP)([^ ]*?)(-[fghilop][^ ]*) (<IN[^>]*>) <(JJ)([^>]*)>}/{\1\2\3 <\1\2-uAP-lI-fNIL \4> <AP-lA-f\5\6>}/ ||
        # branch off final argument embedded question SS w. quotations
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBARQ[^>]*)>}/{\1\2\3 <\1\2-uSS-lI-fNIL \4> <SS-lA-f\5>}/ ||

        #### final NP
        # delete final empty object of passive
        s/{(AP|RP)([^ ]*?)(-[fghilop][^ ]*) (<.*) <NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]>}/<\1\2\3 \4>/ ||
        # delete final *PPA*- empty category
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <[^ ]* \[-NONE- \*PPA\*-[^ ]*\]>}/<\1\2\3 \4>/ ||
        # branch off final IN|TO + NP as modifier RP
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<(?:IN|TO)[^>]*> <NP[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ ||
        # branch off final argument NE
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(WHPP[^ ]* \[IN of\][^>]*)>}/{\1\2\3\4 <\1\2-uNE-lI-fNIL \5> <NE\3-lA-f\6>}/ ||
        # branch off final argument NE
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(PP[^ ]* \[IN of\][^>]*)>}/{\1\2\3\4 <\1\2\3-uNE-lI-fNIL \5> <NE-lA-f\6>}/ ||
        # branch off final argument NE
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) (<IN of> <[^>]*>)}/{\1\2\3\4 <\1\2\3-uNE-lI-fNIL \5> <NE-lA-fNIL \6>}/ ||
        # branch off final argument GP
        s/{([VIBLAGR]P)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(S(?:[-=][^ ]*)?-NOM(?![^ ]*-TMP)[^ ]*) \[NP-SBJ[^ ]* \[-NONE- [^\]]*\]\]([^>]*)>}/{\1\2\3\4 <\1\2-uGP-lI-fNIL \5> <GP\3-lA-f\6\7>}/ ||
        # branch off final argument GS
        s/{([VIBLAGR]P)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(S(?:[-=][^ ]*)?-NOM)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-uGS-lI-fNIL \5> <GS\3-lA-f\6\7>}/ ||
        # branch off final argument NP
        s/{([VIBLAGR]P)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(NP|DT|NN|WHNP|S[^ ]*-NOM)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-uNP-lI-fNIL \5> <NP\3-lA-f\6\7>}/ ||
        # gerund: branch off final argument NP
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(NP|S[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-uNP-lI-fNIL \4> <NP-lA-f\5\6>}/ ||

        # branch off final modifier empty S|SBAR expletive trace (keep around for use in cleft)
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <([^>\]]*-NONE- \*EXP\*[^>\[]*)>}/{\1e\2\3 <\1\2-lI-fNIL \4> <\5>}/ ||

        #### final S
        # branch off final S-ADV with empty subject as modifier RP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE[I]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL <IP-lI-f\5 \6>>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE[A]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5 \6>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE[I]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL <IP-lI-f\5 \6>>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE[A]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5 \6>}/ ||
        # branch off final S with empty subject as argument BP|IP|AP|VP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-u\6P-lI-fNIL \4> <\6P-lA-f\5 \7>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([VIBA])P[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-u\6P-lI-fNIL \4> <\6P-lA-f\5 \7>}/ ||
        # branch off final 'so' + S as modifier RP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) (<[^>\]]* so\]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ ||
        # branch off final IN|TO + S as modifier RP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) (<(?:IN)[^ ]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ ||
        # branch off final IN|TO + S as modifier AP
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<(?:IN|TO)[^>]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lN-fNIL \5>}/ ||
        # branch off final S-ADV as modifier RP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([IA])S[^ ]*(?:-ADV|-PRP)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL <\6S-lI-fS\5>>}/ ||
        # branch off final S as argument VS
        s/{(VE)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([V])S[^>]*)>}/{\1\2\3 <\1\2-u\6S-lM-fNIL \4> <\6S-lI-f\5>}/ ||
        # branch off final ADVP + S as modifier VS|IS|BS|AS
        s/{(SS|EQ|VQ|[VIBLAG]S)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<ADVP[^>]*>) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\7S-lN-fNIL \5 <\7S-lI-f\6>>}/ ||
        # branch off final S as modifier VS|IS|BS|AS
        s/{(SS|EQ|VQ|[VIBLAG]S)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\6S-lN-f\5>}/ ||
        # branch off final S as argument VS|IS|BS|AS
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-u\6S-lI-fNIL \4> <\6S-lA-f\5>}/ ||

        #### final SBAR
        # branch off final SBAR as modifier AP NP (nom clause):                        {NP ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NP <NP ...> <AP <NP WH# ... t# ...>>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) ((?!.*<CC)<.*) <(SBAR[Q]?-NOM)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <NP-lI-f\5\6>>}/ ||
        # branch off final SBAR as modifier AC:                                        {NP ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NP <NP ...> <AC WH# ... t# ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR$SUITABLE_REL_CLAUSE)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AC-oR-lN-f\5>}/ ||
        # branch off final SBAR as argument NP (nom clause):                           {VP ... <SBAR [WH# what/who] ... t# ...>} => {NP <VP-uNP ...> <NP ... t# ...>}
        s/{([VIBLAGR]P)([^ ]*?)(-o[^- ]*)?(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]*-NOM|SBAR [^\]]*(?:when|whether))(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-uNP-lI-fNIL \5> <NP\3-lA-f\6\7>}/ ||
        # branch off final SBAR as argument NP (gerund's nom clause):                  {NP ... <SBAR [WH# what/who] ... t# ...>} => {NP <NP-uNP ...> <NP ... t# ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(SBAR[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-uNP-lI-fNIL \4> <NP-lA-f\5\6>}/ ||
        # branch off final SBAR as argument IP:                                        {VP ... <SBAR [WH# nil] [S [NP nil t#] [ ... to ...]>} => {VP-uIP ... <IP ... to ...>}
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/{\1\2\3 <\1\2-uIP-lI-fNIL \4> <IP-lA-fNIL \6>}/ ||
#        # branch off final SBAR as modifier RP IP:                                    ----> {VP ... <SBAR [WH# nil] [S [NP nil t#] [ ... to ... t# ...]>} => {VP ... <IP ... to ...>}
#        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/{\1\2\3 <\1\2-uIP-lI-fNIL \4> <IP-lA-fNIL \6>}/ ||
#        # branch off final SBAR as modifier RP IP:                                    {VP ... <SBAR [WH# nil] [S [NP nil t#] [ ... to ...]>} => {VP ... <RP [IP ... to ...]>}
#        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/<\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP IP (from SBAR trace coindexed to empty subj)
#        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/<\1\2\3 <\1\2-lI-fNIL \4> <AP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP (from SBAR trace coindexed to final modifier, which must have gap discharged)
#        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]* \[WHADVP(-[0-9]+) [^>]*) \[ADVP[^ ]* \[-NONE- \*T\*\6\]\]([\]]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-f\5\7>}/ ||
        # branch off final SBAR as argument IE-gNP ('tough for X to Y' construction):  {AP ... <SBAR [WH# nil] ... for ... #t ...>} => {AP <AP-uIEg ...> <IE-gNP ... for ... #t ...>}  ****SHOULD BE IE..-lN****
        s/{(AP)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] (\[IN for\] [^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)>}/{\1\2\3 <\1\2-uIEg-lI-fNIL \4> <IE-gNP\5-lM-fNIL \6>}/ ||
        # branch off final SBAR as argument IP-gNP ('tough to Y' construction):        {AP ... <SBAR [WH# nil] ... to ... #t ...>} => {AP <AP-uIPg ...> <IP-gNP ... for ... #t ...>}
        s/{(AP)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[\VP[^ ]* \[TO to\][^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>}/{\1\2\3 <\1\2-uIPg-lI-fNIL \4> <IP-gNP\5-lA-fNIL \6>}/ ||
        # branch off final SBAR as argument AP-gNP ('worth Y-ing' construction):       {AP ... <SBAR [WH# nil] ... #t ...>} => {AP <AP-uAPg ...> <AP-gNP ... #t ...>}
        s/{(AP)([^ ]*?)(-[fghilop][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>}/{\1\2\3 <\1\2-uAPg-lI-fNIL \4> <AP-gNP\5-lA-fNIL \6>}/ ||
        # branch off final SBAR as modifier IP-gNP (NP_i [for X] to find pics of t_i): {NP ... <SBAR [WHNP# nil] ... #t ...>} => {NP <AP ...> <IP-gNP ... #t ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]*) \[WHNP[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>}/{\1\2\3 <\1\2-lI-fNIL \4> <IP-gNP\6-lN-f\5 \7>}/ ||
        # branch off final SBAR as modifier IP-gRP (NP_i [for X] to say you ... t_i):  {NP ... <SBAR [WH# nil] ... #t ...>} => {NP <AP ...> <IP-gRP ... #t ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]*) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>}/{\1\2\3 <\1\2-lI-fNIL \4> <IP-gRP\6-lN-f\5 \7>}/ ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# where/when] ... t# ...>} => {NP <VP-uNP ...> <NP ... t# ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-lN-f\5\6>}/ ||
        # branch off final SBAR as modifier AC:                                        {NP ... <SBAR [WH# where/when] ... t# ...>} => {NP <VP-uNP ...> <NP ... t# ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AC-lN-f\5\6>}/ ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# which] ... t# ...>} => {VP <VP ...> <RC WH# ... t# ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*which[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-oR-lN-f\5\6>}/ ||
        # branch off final SBAR as modifier AC (that|nil):                             {NP ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {VP <VP ...> <AC WH# ... t# ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AC-oR-lN-f\5\6>}/ ||
        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-uQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR(?!-ADV|-TMP)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that|[^\]]*-NONE-))([^>]*)>}/{\1\2\3 <\1\2-uEQ-lI-fNIL \4> <EQ-lA-f\5\6>}/ ||
#        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+(?![^\]]*what|[^\]]*why)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-lN-f\5\6>}/ ||
        # branch off final SBAR as modifier RP colon:                                  {VP ... <:> <SBAR [IN because/..] ...>} => {VP <VP ...> <AP <:> <AP because ...>>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*<SBAR[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ ||
        # branch off final SBAR as modifier AP colon:                                  {NP ... <:> <SBAR [IN because/..] ...>} => {NP <NP ...> <AP <:> <AP because ...>>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<[^ ]* :> <.*<SBAR[^>]*)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL \5>}/ ||
        # branch off final SBAR as modifier IE:                                        {NP ... <SBAR [IN because/..] ...>} => {NP <NP ...> <AP ...>}
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]* \[[^ ]* for\] [^>]*)>}/{\1\2\3 <\1\2-uIE-lI-fNIL \4> <IE-lA-f\5>}/ ||
        # branch off final SBAR as modifier RP:                                        {VP ... <SBAR [IN because/..] ...>} => {VP <VP ...> <AP ...>}
#WORSE:        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*-TMP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-t-lM-f\5\6>}/ ||
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*(?:-ADV|-LOC|-TMP|-CLR)|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5\6>}/ ||
        # branch off final SBAR as modifier AP:                                        {NP ... <SBAR [IN because/..] ...>} => {NP <NP ...> <AP ...>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[Q]?-LOC|SBAR[Q]?-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ ||
        # branch off final SBAR as argument AP NP (nom clause following 'being'):      {NP ... being ... <SBAR [WH# what/who] ... t# ...>} => {NP <NP-uAP ... being ...> <AP <NP ... t# ...>>}
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) (.* being>.*) <(SBAR-NOM|SBAR-PRD)([^>]*)>}/{\1\2\3 <\1\2-uAP-lM-fNIL \4> <AP-lI-fNIL <NP-lI-f\5\6>>}/ ||
#        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-uQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
#        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)>}/{\1\2\3 <\1\2-uEQ-lI-fNIL \4> <EQ-lA-f\5\6>}/ ||
        # delete final empty SBAR given gap AC and re-run:
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP]|NP|NN)(-gAC)(-[0-9]+)([^ ]*) (<.*) <SBAR[^>\]]*\[-NONE- \*(?:T|ICH)\*\3\][^>\[]*>}/<\1\2\3\4 \5>/ ||
        # branch off final SBAR as argument VE:                                        {XP ... <SBAR [IN that/nil] ...>} => {XP <XP-uVE ...> <VE ...>}   *****SHOULD BE VE******
        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SBAR)([^>]*)>}/{\1\2\3 <\1\2-uVE-lI-fNIL \4> <VE-lA-f\5\6>}/ ||

        #### final RP
        # branch off final modifier RP colon
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ ||
        # branch off final modifier RP
        s/{(SS|EQ|VQ|VE|IE|[VIBLAGR][SP])([^ ]*?)(-[fghilop][^ ]*) (<.*) <(?![^ ]*-PRD)(RB|ADVP|PP|UCP|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5\6>}/ ||
        # branch off final SQ|SINV as argument SS
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(SQ|SINV)([^>]*)>}/{\1\2\3 <\1\2-uSS-lI-fNIL \4> <SS-lA-f\5\6>}/ ||
        # branch off final INJP as argument VE
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(INTJ)((?![^ ]*-ADV)[^>]*)>}/{\1\2\3 <\1\2-uVE-lI-fNIL \4> <VE-lA-f\5\6>}/ ||
        # branch off final argument LS ('had I only known' construction)
        s/{([VIBLAGR]P)([^ ]*?)(-[fghilop][^ ]*) (<VBD had>) (.*<NP.*<VP.*)}/{\1\2\3 <\1\2-uLS-lM-fNIL \4> <LS-lI-fNIL \5>}/ ||

        #### final AP
#        # semi/dash splice
#        s/{(NP)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<[^ ]* ;>) (<.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <C\1 \5 <\1\2-lI-fNIL \6>>}/ ||
        # gerund: delete initial empty subject (passing -o to head)
        s/{(NP)([^ ]*?)(-[fghilp][^ ]*) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)}/<\1\2\3 \4>/ ||
        # branch off final modifier AP appositive NP (passing -o to head)
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) ((?=.*<(?:NP|[^ ]*-NOM).*<(?:NP|[^ ]*-NOM))(?!.*<CC)<.*) <(NP|[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <NP-lI-f\5\6>>}/ ||
        # branch off final modifier AP infinitive phrase (with TO before any VBs) (passing -o to head)  ****************
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) (<.*) <(VP(?:(?!\[VB)[^>])*\[TO[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <IP-lI-f\5>>}/ ||
        # branch off final modifier AP (passing -o to head)
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) (<.*) <(RRC|PP|ADJP|ADVP|RB|UCP|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ ||
        # branch off final modifier AP (from VP) (passing -o to head)
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) (.*<(?:NN|NP).*) <(VP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ ||
        # branch off final argument LP (passing -o to head)
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) (.* having>.*) <(VP)([^>]*)>}/{\1\2\3 <\1\2-uLP-lM-fNIL \4> <LP-lI-f\5\6>}/ ||
        # branch off final argument EQ (passing -o to head)  **WH EQ WILL NEVER FIRE**
        s/{(NP|NN)([^ ]*?)(-[fghilp][^ ]*) (<.*) <(SQ(?=[^ ]* \[WH))([^>]*)>}/{\1\2\3 <\1\2-uEQ-lI-fNIL \4> <EQ-lA-f\5\6>}/ ||

        #### final misc needed by SS   *****COULD BE GENERALIZED******
        # branch off final 'so' + S
        s/{(SS|EQ|VQ|VE|IE|[VIBLAG]S)([^ ]*?)(-[fghilop][^ ]*) (<.*) (<[^ ]* so> <S(?![A-Z])[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ ||

        ######################################################################
        ## 5. LOW PRECEDENCE BRANCHING INITIAL CONSTITUENTS

        #### VS
        # inverted declarative sentence: branch off final subject
        s/{([VIBLAG])S([^ ]*?)(-[fghilop][^ ]*) (<.*) <(NP[^ ]*-SBJ)([^>]*)>}/{\1S\2\3 <\1P\2-lI-fNIL \4> <NP-lA-f\5\6>}/ ||
        # [VIBLAG] sentence: branch off initial VE subject
        s/{([VIBLAG])S([^ ]*?)(-[fghilop][^ ]*) <(SBAR[^ ]*-SBJ[^\]]*that)([^>]*)> (<.*)}/{\1S\2\3 <VE-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||
#        # [VIBLAG] sentence: branch off initial EQ subject
#        s/{([VIBLA])S([^ ]*?)(-[fghilop][^ ]*) <(SBAR(?!-ADV)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)> (<.*)}/{\1S\2\3 <EQ-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||
        # [VIBLAG] sentence: branch off initial IE subject
        s/{([VIBLAG])S([^ ]*?)(-[fghilop][^ ]*) <(SBAR[^ ]*-SBJ[^\]]*for)([^>]*)> (<.*)}/{\1S\2\3 <IE-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||
        # [VIBLAG] sentence: branch off initial IP subject
        s/{([VIBLAG])S([^ ]*?)(-[fghilop][^ ]*) <(SheadI[^ ]*-SBJ)([^>]*)> (<.*)}/{\1S\2\3 <IP-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||
        # [VIBLAG] sentence: branch off initial NP subject
        s/{([VIBLAG])S([^ ]*?)(-[fghilop][^ ]*) <(NP|[^ ]*-NOM|[^ ]*-SBJ)([^>]*)> (<.*)}/{\1S\2\3 <NP-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||

        #### NP
        # branch off initial punctuation
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) <([^ ]*) ($INIT_PUNCT)> (<.*)}/{\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>}/ ||
        # branch off initial parenthetical sentence with extraction
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)}/{\1\2\3 <VS-gSS\6-lN-f\4\5> <\1\2-lI-fNIL \7>}/ ||
        # branch off initial pre-determiner
        s/{(NP)([^ ]*?)(-o[^- ]*)?([^ ]*?)(-[fghilop][^ ]*) <(PDT|RB)([^>]*)> (<.*)}/{\1\2\3\4\5 <DD-lM-f\6\7> <NP\2\3\4-lI-fNIL \8>}/ ||
        # branch off initial determiner (wh)
        s/{(NP)([^ ]*?)(-o[^- ]*)?([^ ]*?)(-[fghilop][^ ]*) <(WDT|WP\$)([^>]*)> (<.*)}/{\1\2\3\4\5 <DD\3-lM-f\6\7> <NN\2\4-lI-fNIL \8>}/ ||
        # branch off initial determiner (non-wh)
        s/{(NP)([^ ]*?)(-o[^- ]*)?([^ ]*?)(-[fghilop][^ ]*) <(DT|PRP\$|PRP|NP[^>]*\[POS 's?\])([^>]*)> (<.*)}/{\1\2\3\4\5 <DD-lM-f\6\7> <NN\2\3\4-lI-fNIL \8>}/ ||
        # branch off initial modifier AA
        s/{(NP|NN)([^ ]*?)(-o[^- ]*)?([^ ]*?)(-[fghilop][^ ]*) <(WHADJP|WRB)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3\4\5 <AA\3-lM-f\6\7> <NN\2\4-lI-fNIL \8>}/ ||
        # branch off initial modifier AA
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) <(NN|CD|QP|JJ|ADJP|WHADJP|IN|PP|RB|TO|ADVP|VB|UCP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3 <AA-lM-f\4\5> <NN\2-lI-fNIL \6>}/ ||
        s/{(NP|NN)([^ ]*?)(-[fghilop][^ ]*) <(NAC)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3 <AA-lM-fNIL <NP-lI-f\4\5>> <NN\2-lI-fNIL \6>}/ ||
        # rebinarize QP containing dollar sign followed by *U*, and continue
        s/{(NP|NN|AA)([^ ]*?)(-[fghilop][^ ]*) <QP ([^>]*)\[([^ ]* [^ ]*[\$\#][^ ]*)\] ([^>]*)>(?: <-NONE- \*U\*>)?}/<\1\2\3 \4\[\5\] \[QP \6\]>/ ||
        # branch off currency unit followed by non-final *U*
        s/{(NP|NN|AA)([^ ]*?)(-[fghilop][^ ]*) (<[^ ]* [\$\#][^ ]*>.*) <-NONE- \*U\*> (<.*)}/{\1\2\3 <AA-lM-fNIL \4> <NN-lI-fNIL \5>}/ ||
        # rebinarize currency unit followed by QP
        s/{(NP|NN|AA)([^ ]*?)(-[fghilop][^ ]*) <([^ ]* [\$\#][^ ]*)> (.*?)( <-NONE- \*U\*>)?}/{\1\2\3 <\$-lI-f\4> <AA-lM-fNIL \5>}/ ||

        #### AA|RR|NN
        # branch off initial modifier RR
        s/{(AA)([^ ]*?)(-o[^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <RR\3-lM-f\5> <AA-lI-fNIL \6>}/ ||
        s/{(AA)([^ ]*?)(-o[^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <RR-lM-f\5> <AA\3-lI-fNIL \6>}/ ||
#        s/{(AA)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <RR-lM-f\3> <AA-lI-fNIL \4>}/ ||
        s/{(RR)([^ ]*?)(-o[^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <RR\3-lM-f\5> <RR-lI-fNIL \6>}/ ||
        s/{(RR)([^ ]*?)(-o[^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <RR-lM-f\5> <RR\3-lI-fNIL \6>}/ ||
#        s/{(RR)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <RR-lM-f\3> <RR-lI-fNIL \4>}/ ||
        s/{(NN)([^ ]*?)(-o[^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <AA\3-lM-f\5> <NN-lI-fNIL \6>}/ ||
        s/{(NN)([^ ]*?)(-o[^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <AA-lM-f\5> <NN\3-lI-fNIL \6>}/ ||
#        s/{(NN)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <AA-lM-f\3> <NN-lI-fNIL \4>}/ ||
        s/{(CC)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <CC-lM-f\3> <CC-lI-fNIL \4>}/ ||

        #### panic as MWE
#        s/{([VIBLAGR]P|NP|NN)([^ ]*?)(-[fghilop][^ ]*) (<.*) <(?!-NONE-)([^ ]*) (..?.?.?)>}/{\1\2\3 <\1\2-uAP\6-fNIL \4> <AP\6-lA-f\5 \6>}/ ||

        0 ) {}

    ######################################################################
    ## C. ADJUST BRACKETS OF CONSTITUENTS WITHIN NEWLY-INSERTED CONSTITUENTS

    while (
           #### uh-oh, turn each minimal <...> pair within other <...> into [...]
           s/(<[^>]*)<([^<>]*)>/\1\[\2\]/ ||
           0 ) {}

    ######################################################################
    ## D. PROPAGATE GAPS AND RELATED TAGS DOWN

    while (
	   debug($step, "?? $_") ||

           #### -g/h/i (gap/trace tags)
           # propagate -g gap tag from parent to each child dominating trace with matching index
           s/{([^ ]*)(-g[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghilop][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)}/{\1\2\3\4 <\5\2\3\6>\7}/ ||
           # propagate -h right-node-raising gap tag from parent to each child dominating trace with matching index
           s/{([^ ]*)(-h[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghilop][^>]*\[-NONE- *\*(?:RNR)\*\3\][^>]*)>(.*)}/{\1\2\3\4 <\5\2\3\6>\7}/ ||
           # add -i it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-i)([^ ]*?)(-[fghilop][^>]*\[[^ ]*\2[^>]*)>/\1<\3-iNPe\4>/ ||
           # add -s spec tag to sibling of cleft complement following expletive trace (-s then passed down thru syn heads)
           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-s)([^ ]*?)(-[ufghilop][^>]*)> <([^ ]*\2[^>]*)>/\1<\3-sNPe\4> <\5>/ ||
           # add -s spec tag to sibling of cleft complement containing expletive trace (-s then passed down thru syn heads)
           s/<(?![^ ]*-s)([^ ]*?)(-[ufghilop][^>]*-NONE- \*EXP\*(-[0-9]+)[^0-9][^>]*)> <([^ ]*\3[^>]*)>/<\1-sNPe\2> <\4>/ ||
#           # turn last -i tag into -s
#           s/{([^ ]*)-i(?!.*<[^ ]*-i.*\})(.*)}/{\1-s\2}/ ||
##           # add -i it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)\{(?![^ ]*-i)([^ ]*?)(-[fghilop].*\[[^ ]*\2.*)\}/\1\{\3-iNPe\4\}/ ||
##           # turn last -i tag into -s
##           # add -s spec tag to each constituent following expletive trace that immediately dominates cleft complement with matching index (-s then passed down thru syn heads)
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-s)([^ ]*?)(-[fghilop].*\[[^ ]*\2.*)>/\1<\3-sNPe\4>/ ||
           0 ) {}

    ####################
    ## mark current as external...
    s/{(.*?)}/\(\1\)/;
    ## mark first unexpanded child as current...
    s/<(.*?)>/{\1}/;
  }

  ######################################################################
  ## III. REMOVE TEMPORARY ANNOTATIONS: EMPTY CATEGORIES, TRACE NUMBERS, ETC.

#  # delete all empty cats
#  while ( 
#         # empty category deletion (quad)
#         s/ \([^ ]* \([^ ]* \([^ ]* \([^ ]*-NONE- [^\)]*\)\)\)\)//g ||
#         # empty category deletion (triple)
#         s/ \([^ ]* \([^ ]* \([^ ]*-NONE- [^\)]*\)\)\)//g ||
#         # empty category deletion (double)
#         s/ \([^ ]* \([^ ]*-NONE- [^\)]*\)\)// ||
#         # empty category deletion (single)
#         s/ \([^ ]*-NONE- [^\)]*\)// ||
#         0 ) {}
  # delete only empty cats for gaps with corresponding fillers
  while (
         # empty category deletion (quad)
         s/ \([^ ]*-[gh][^ ]*(-[0-9])+[^ ]* \([^ ]* \([^ ]* \([^ ]*-NONE- \*(?:T|ICH|RNR|INTERNAL)\*\1[^\)]*\)\)\)\)//g ||
         # empty category deletion (triple)
         s/ \([^ ]*-[gh][^ ]*(-[0-9])+[^ ]* \([^ ]* \([^ ]*-NONE- \*(?:T|ICH|RNR|INTERNAL)\*\1[^\)]*\)\)\)//g ||
         # empty category deletion (double)
         s/ \([^ ]*-[gh][^ ]*(-[0-9])+[^ ]* \([^ ]*-NONE- \*(?:T|ICH|RNR|INTERNAL)\*\1[^\)]*\)\)// ||
         # empty category deletion (single)
         s/ \([^ ]*-[gh][^ ]*(-[0-9])+[^ ]*-NONE- \*(?:T|ICH|RNR|INTERNAL)\*\1[^\)]*\)// ||
         0 ) {}

  # remove trace numbers from gaps
  s/(-g[A-Z]+)-[0-9]+/\1/g;
  # turn right-node-raising gaps into complement tags
  s/-h([A-Z]+)-[0-9]+/-u\1/g;
  # turn expletive it-cleft traces into specifier tags
  s/-i([A-Z]+)/-s\1/g;

  # correct VBNs
  s/\(LP([^ ]*)-fVB[A-Z]*/\(LP\1-fVBN/g;
  s/\((AP|AA|RP|RR)([^ ]*)-fVB[ND]/\(\1\2-v-fVBN-v/g;     ## mark passive
  s/\((AP|AA|RP|RR|NP|NN)([^ ]*)-fVB[G]/\(\1\2-r-fVBG-r/g;   ## mark progressive/gerund

#  s/\(([^ ]*)(-fVB)/\(\1-wasvb\2/g;  ## helps mecommon?

#  # throw out new categories
#  s/[^ ]+-f//g;
#  s/\(VBN-v ([^\)]*)\) *\(NP[^ ]* *\(-NONE- [^\)]*\) *\)/\(VBN-v \1\)/g;
#  s/\(VP \(VBN-v ([^\)]*)\) *\)/\(VP-v \(VBN-v \1\)\)/g;

#  # throw out old categories
#  s/\(([^ ]*)-f[^ ]*/\(\1/g;

# put the synrole -l at the end
#  s/-l([A-Z])([^ ]*)/\2-l\nn1/g;

  # clean up -TOBE..
  s/-TOBE..//g;

  # add unary branch to generalize gap categories
  s/\(([^ ]*)(-l[^I])([^ ]*) ([^\(\)]*)\)/\(\1\2\3 \(\1-lI\3 \4\)\)/g;
  s/\(([^ ]*)-g(NP|SS)([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1-u\2\3 \4\)\)/g;
  s/\(([^ ]*)-g(RP|AC)([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1\3 \4\)\)/g;

  # output
  print $_;
}
