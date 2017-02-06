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
  s/^\(NP(?!-f)/\(NS-fNP/;
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
           s/{([^ ]*-f)[^ ]* <NP(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>}/<\1NS\2>/ ||
           s/{([^ ]*-f)[^ ]* <NN(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>}/<\1NP\2>/ ||
           s/{([^ ]*-f)[^ ]* <(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>}/<\1\2>/ ||
           0 ) {}


    ######################################################################
    ## B. BRANCH OFF MODIFIERS AND ARGUMENTS IN ORDER OF PRECEDENCE

    #### only one rewrite rule may apply at any node (recurse only if new node created)
    #### rewrite rules are ordered by precedence: rules preferred at higher nodes are first
    $j = 0;
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

        #### remove empty NS from S turned to [VIBLA]P
        (s/{([VIBLAR]P[^ ]*.*) <NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]>(.*)}/<\1\2>/ && ($j=1)) ||

        #### parentheticals
        # identify ptb tag dominating its own trace, and change it to *INTERNAL*
        (s/{([^ ]*)(-f[^ ]*)(-[0-9]+)((?![0-9]).*-NONE- \*)(?:T|ICH)(\*\3(?![0-9]).*)}/<\1\2\3\4INTERNAL\5>/ && ($j=2)) ||
        # flatten PRN nodes
        (s/{(.*) <PRN([^>]*)>(.*)}/<\1\2\3>/ && ($j=3)) ||

        ######################################################################
        ## 2. HIGH PRECEDENCE BRANCHING FINAL CONSTITUENT

        #### SS
        # semi/dash splice between matching constituents
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) <([^- ]*)([^>]*)> <([^ ]*) (;|--)> <\4([^>]*)>}/{\1\2\3 <\1\2-lC-f\4\5> <Cs\1-lI-fNIL <\6-lI-f\6 \7> <\1\2-lC-f\4\8>>}/ && ($j=4)) ||
#        # inverted sentence: branch off final raised complement SS (possibly quoted) with colon
#        s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- *\*ICH\*(-[0-9]+)(?![0-9]).*) <: :> <(S)([^ ]*)\5([^>]*)>}/{\1\2\3 <VS-hSS\5\2-lI-fNIL \4> <SSmod-lI-f\6\7 <: :> <SS-lI-f\6\7\5\8>>}/ ||
        # branch off final SBAR as extraposed modifier ZSr rel clause:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gZSr ...> <ZSr WH# ... t# ...>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 $SUITABLE_REL_CLAUSE)>}/{\1\2\3 <\1\2-gZR\5-lI-fNIL \4> <ZSr-rNS-lN-f\6>}/ && ($j=5)) ||
        # branch off final SBAR as extraposed modifier IP:                           {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gRP ...> <IP WH# ... t# ...>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*) \[ADVP \[-NONE- \*T\*\7\]\]([^>]*)\]>}/{\1\2\3 <\1\2-gZR\5-lI-fNIL \4> <IP-lN-f\6 \8\9>}/ && ($j=6)) ||
        # branch off final SBAR as extraposed modifier ZS complement:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gZR ...> <ZS ...>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 \[IN that\][^>]*)>}/{\1\2\3 <\1\2-gZS\5-lI-fNIL \4> <ZS-lN-f\6>}/ && ($j=7)) ||
        # inverted sentence: branch off final raised complement SS (possibly quoted)
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(S)([^ ]*)\5([^>]*)>}/{\1\2\3-modeverused? <VS-gSS\5\2-lI-fNIL \4> <SS-lN-f\6\7\5\8>}/ && ($j=8)) ||

        # branch off final punctuation (passing -o to head)
        (s/{(NS)([^ ]*?)(-[fghjlp][^ ]*?)NP (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>}/{\1\2\3NS <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>}/ && ($j=8.5)) ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|ZSr|RC|[VIBLAGR][SP]|AI|RI|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>}/ && ($j=9)) ||

        # branch off final possessive 's
        (s/{(DD|NS)([^ ]*) (<.*) <(POS 's?)>}/{\1\2 <NS-lA-fNIL \3> <DD-aNS-lI-f\4>}/ && ($j=10)) ||

        ######################################################################
        ## 3. HIGH PRECEDENCE BRANCHING INITIAL CONSTITUENTS

        # branch off initial punctuation (passing -o to head)
        (s/{(SS|VS\-iNS|QS|ZS|ES|ZSr|RC|[VIBLAGR][SP]|NS|AI|RI|NP)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*) ($INIT_PUNCT)> (<.*)}/{\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>}/ && ($j=11)) ||

        #### AP|RP
        # unary expand to NS
        (s/{(AP|RP)([^ ]*?)(-[g][^ ]*)?-fN[P|S]([^ ]*) (<.*)}/{\1\2\3-fNS\4 <NS\2-lI-fNIL \5>}/ && ($j=12)) ||
        # unary expand to NS (nom clause)
        (s/{(AP)([^ ]*?)(-[g][^ ]*)?(-fSBAR[^ ]*(?= <WH)) (<.*)}/{\1\2\3\4 <NS\2-lI-fNIL \5>}/ && ($j=13)) ||
        # branch off initial specifier NS measure
        (s/{(AP|RP)([^ ]*?)(-[fghjlp][^ ]*) <NP([^>]*)> (<.*)}/{\1\2\3 <NS-lA-fNS\4> <\1\2-aNS-lI-fNIL \5>}/ && ($j=14)) ||
        # delete initial empty subject
        (s/{(RP)([^ ]*?)(-[fghjlp][^ ]*) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)}/<\1\2\3 \4>/ && ($j=15)) ||
#        # branch off initial modifier RI
#        s/{(RP)([^ ]*?)(-[fghjlp][^ ]*) <(RB|ADVP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3 <RI-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
#        s/{(AP|RP)([^ ]*?)(-[fghjlp][^ ]*) <(RB|ADVP|ADJP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO).*)}/{\1\2\3 <RI-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
##           s/{(AP|RP)([^ ]*)(-f[^ ]*) (<TO.*<VP.*)}/{\1\2\3 <IP\2-fNIL \4>}/ ||
        # for good / for now
        (s/{(RP)([^ ]*?)(-[fghjlp][^ ]*) <(IN[^>]* for)> <(?![^>]* long[^>]*)(?![^>]* awhile[^>]*)(RB[^>]*|ADVP[^ ]* \[RB[^>]*)>}/{\1\2\3 <RP-bNS-lI-f\4> <NS-lA-f\5>}/ && ($j=16)) ||

        #### initial filler
        # content question: branch off initial interrogative NS
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NS-iNS-lN-f\4\5\6\7> <QS\2-gNS\6-lI-fNIL \8>}/ && ($j=17)) ||
        # content question: branch off initial interrogative RP
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-iNS-lN-f\4\5\6\7> <QS\2-gRP\6-lI-fNIL \8>}/ && ($j=18)) ||
        # topicalized sentence: branch off initial topic SS (possibly quoted)
        (s/{(SS|ZS|ES|VS)([^ ]*?)(-[fghjlp][^ ]*) (?!<[^ ]*-SBJ)<(S)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <SS-lN-f\4\5\6\7> <VS\2-gSS\6-lI-fNIL \8>}/ && ($j=19)) ||
        # topicalized sentence: branch off initial topic NS   ***<[^ ]* \[-NONE- [^\]]*\]>|
        (s/{(SS|ZS|ES|VS)([^ ]*?)(-[fghjlp][^ ]*) (?!<[^ ]*-SBJ)<(NP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NS-lN-f\4\5\6\7> <VS\2-gNS\6-lI-fNIL \8>}/ && ($j=20)) ||
        # topicalized sentence: branch off initial topic AP
        (s/{(SS|ZS|ES|VS)([^ ]*?)(-[fghjlp][^ ]*) (?!<[^ ]*-SBJ)<(ADJP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <AP-lN-f\4\5\6\7> <VS\2-gAP\6-lI-fNIL \8>}/ && ($j=21)) ||
        # topicalized sentence: branch off initial topic RP
        (s/{(SS|ZS|ES|VS)([^ ]*?)(-[fghjlp][^ ]*) (?!<[^ ]*-SBJ)<((?!WH)[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-lN-f\4\5\6\7> <VS\2-gRP\6-lI-fNIL \8>}/ && ($j=22)) ||
        # embedded sentence: delete initial empty complementizer
        (s/{(ZS)([^ ]*?)(-[fghjlp][^ ]*) <-NONE-[^>]*> (<.*)}/{\1\2\3 <VS\2-lI-fNIL \4>}/ && ($j=23)) ||
        # embedded sentence: branch off initial complementizer
#        s/{(V|I)E([^ ]*?)(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{\1E\2\3 <\1E\2-b\1S-lM-f\4> <\1S-lI-fNIL \5>}/ ||
        (s/{ZS(?!r)([^ ]*?)(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{ZS\1\2 <ZS\1-bVS-lM-f\3> <VS-lI-fNIL \4>}/ && ($j=24)) ||
        (s/{ES([^ ]*?)(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{ES\1\2 <ES\1-bIS-lM-f\3> <IS-lI-fNIL \4>}/ && ($j=25)) ||
        # embedded noun: branch off initial preposition
#        s/{(N)E([^ ]*?)(-[ir][A-Z])?(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{\1E\2\3\4 <\1E\2-b\1P-lM-f\5> <\1P\3-lI-fNIL \6>}/ ||
        (s/{OS([^ ]*?)(-[ir][A-Z]+)?(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{OS\1\2\3 <OS\1-bNS-lM-f\4> <NS\2-lI-fNIL \5>}/ && ($j=26)) ||
        # embedded question: branch off initial interrogative RP whether/if
        (s/{(VS\-iNS)([^ ]*?)(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{\1\2\3 <RP-iNS-lI-f\4> <VS\2-gRP-lN-fNIL \5>}/ && ($j=27)) ||

        #### initial RP/RC modifier
        # branch off initial modifier RP with colon
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S)([^ ]*?)(-[fghjlp][^ ]*) <(PP|RB|ADVP|CC|FRAG|NP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> <([^ ]*) (:)> (<.*)}/{\1\2\3 <RP-lM-fNIL <RP-lI-f\4\5> <\6 \7>> <\1\2-lI-fNIL \8>}/ && ($j=28)) ||
        # branch off initial modifier RP IP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <S[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ($NO_AP_HEAD\[TO[^>]*)> (<.*)}/{\1\2\3 <RP-lM-fNIL <IP-lI-fNIL \4>> <\1\2-lI-fNIL \5>}/ && ($j=29)) ||
        # branch off initial modifier RP AP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <S[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ([^>]*)> (<.*)}/{\1\2\3 <RP-lM-fNIL \4> <\1\2-lI-fNIL \5>}/ && ($j=30)) ||
        # branch off initial modifier RP from SBAR
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <(SBAR(?![^ ]*-SBJ)[^ ]* (?!\[IN that|\[IN for|\[IN where|\[IN when)(?!\[WH[^  ]*))([^>]*)> (<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ && ($j=31)) ||
        # branch off initial modifier RC from SBAR-ADV
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <(SBAR(?:-ADV|-TMP)[^ ]* \[WH)([^>]*)> (<.*)}/{\1\2\3 <RC-rNS-lN-f\4\5> <\1\2-lI-fNIL \6>}/ && ($j=32)) ||
        # branch off initial RB + JJS as modifier RP  (e.g. "at least/most/strongest/weakest")
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|[VIBLG]P|NS)([^ ]*?)(-[fghjlp][^ ]*) (<IN[^>]*> <JJ[^>]*>) (?!<CC)(<.*)}/{\1\2\3 <RP-lM-fNIL \4> <\1\2-lI-fNIL \5>}/ && ($j=33)) ||
        # branch off initial modifier RP  (incl determiner, e.g. "both in A and B")
#WORSE:        s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*-TMP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3 <RP-t-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        (s/{(SS|VS\-iNS|QS|ZS(?!r)|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <(NP-TMP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3 <RP-lM-fNS-TMP\5> <\1\2-lI-fNIL \6>}/ && ($j=33.5)) ||
        (s/{(SS|VS\-iNS|QS|ZS(?!r)|ES|[VIBLAG]S|[VIBLG]P)([^ ]*?)(-[fghjlp][^ ]*) <(DT|PP|RB|IN|ADVP|CC|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ && ($j=34)) ||
        (s/{(NS)([^ ]*?)(-[ir][A-Z]+)?([^ ]*?)(-[fghjlp][^ ]*) <(CC)([^>]*)> (<(?!PP|WHPP).*)}/{\1\2\3\4\5 <RI-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ && ($j=35)) ||
        (s/{(NS)([^ ]*?)(-[ir][A-Z]+)?([^ ]*?)(-[fghjlp][^ ]*) <(RB|PDT|(?:CC|DT) (?:[Nn]?[Ee]ither|[Bb]oth)(?=.*<CC.*\})|DT (?:[Aa]ll|[Bb]oth|[Hh]alf)|(?:ADJP|QP)(?=[^>]*\[DT [^\]]*\]>))([^>]*)> (<.*)}/{\1\2\3\4\5 <RI-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ && ($j=36)) ||
        (s/{(NS)([^ ]*?)(-[ir][A-Z]+)?([^ ]*?)(-[fghjlp][^ ]*) <(ADVP|PP)([^>]*)> (?!<CC)(<.*)}/{\1\2\3\4\5 <RP-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>}/ && ($j=37)) ||
#        s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)(-[fghjlp][^ ]*) <(S)([^ ]* \[NP[^ ]* \[-NONE- \*-[^>]*)> (<.*)}/{\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>}/ ||
        # branch off initial modifier RI-iNS/R of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        (s/{(AP|RP)([^ ]*?)(-[ir][A-Z]+)(-[fghjlp][^ ]*) <(WRB)([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3\4 <RI\3-lM-f\5\6> <\1\2-lI-fNIL \7>}/ && ($j=38)) ||
        # branch off initial modifier RI of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        (s/{(AP|RP)([^ ]*?)(-[fghjlp][^ ]*) <(DT|PP|RB|IN(?=[^\|]*<IN)|ADVP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)}/{\1\2\3 <RI-lM-f\4\5> <\1\2-lI-fNIL \6>}/ && ($j=39)) ||

        #### CODE REVIEW: WHADVP/WP$ in NS needs to inherit -o  ******************

        #### sentence types
        # branch off initial parenthetical sentence with extraction
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)}/{\1\2\3 <VS-gSS\6-lN-f\4\5> <\1\2-lI-fNIL \7>}/ && ($j=40)) ||
        # branch off initial parenthetical sentence w/o extraction
        (s/{(VP)([^ ]*?)(-[fghjlp][^ ]*) <(S(?![A-Z]))([^>]*)> (.*<VP.*)}/{\1\2\3 <VS-lN-f\4\5> <\1\2-lI-fNIL \6>}/ && ($j=41)) ||
        # imperative sentence: delete empty NS
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) <NP[^ ]* \[-NONE- \*\]> (<VP$NO_VP_HEAD \[VB.*)}/{\1\2\3 <BP\2-lI-fNIL \4>}/ && ($j=42)) ||
        # declarative (inverted or uninverted) sentence: unary expand to VS
        (s/{(SS|ZS)([^ ]*?)(-[fghjlp][^ ]*) (<(?:NP|[^ ]*-SBJ).*<VP.*|<VP.*<(?:NP|[^ ]*-SBJ))}/{\1\2\3 <VS\2-lI-fNIL \4>\5}/ && ($j=43)) ||
        # polar question: unary expand to QS
        (s/{(SS|VS\-iNS)([^ ]*?)(-[fghjlp][^ ]*) (<[^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Hh]as|[Hh]ave|[Hh]ad)).*<NP.*)}/{\1\2\3 <QS\2-lI-fNIL \4>}/ && ($j=44)) ||
        # imperative sentence: unary expand to BP   ***PROBABLY NULL CAT HERE***
        (s/{(SS)([^ ]*?)(-[fghjlp][^ ]*) (<VP$NO_VP_HEAD \[VB.*)}/{\1\2\3 <BP\2-lI-fNIL \4>}/ && ($j=45)) ||
        # embedded question / nom clause: branch off initial interrogative NS and final modifier IP with NS gap (what_i to find a picture of t_i)
        (s/{(VS\-iNS|NS)([^ ]*?)(-[fghjlp][^ ]*) <(WHNP[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>}/{\1\2\3 <NS-iNS-lI-f\4\5\6> <IP-gNS\5-lN-fNIL \7>}/ && ($j=46)) ||
        # embedded question / nom clause: branch off initial interrogative RP and final modifier IP with RP gap (how_i to find a picture t_i)
        (s/{(VS\-iNS|NS)([^ ]*?)(-[fghjlp][^ ]*) <(WH[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>}/{\1\2\3 <RP-iNS-lI-f\4\5\6> <IP-gRP\5-lN-fNIL \7>}/ && ($j=47)) ||
        # embedded question / nom clause: branch off initial interrogative NS
        (s/{(VS\-iNS|NS)([^ ]*?)(-[fghjlp][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <NS-iNS-lI-f\4\5\6\7> <VS\2-gNS\6-lN-fNIL \8>}/ && ($j=48)) ||
        # embedded question / nom clause / nom clause modifier: branch off initial interrogative RP
        (s/{(VS\-iNS|NS)([^ ]*?)(-[fghjlp][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2\3 <RP-iNS-lI-f\4\5\6\7> <VS\2-gRP\6-lN-fNIL \8>}/ && ($j=49)) ||
        # polar question: branch off initial BP-taking auxiliary
        (s/{(QS)([^ ]*?)(-[fghjlp][^ ]*) <([^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|'d))[^>]*)> (<.*)}/{\1\2\3 <\1\2-bBS-lM-f\4> <BS-lI-fNIL \5>}/ && ($j=50)) ||
        # polar question: branch off initial NS-taking auxiliary
        (s/{(QS)([^ ]*?)(-[fghjlp][^ ]*) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<(?:NP|NN|DT)[^>]*>)}/{\1\2\3 <\1\2-bNS-lM-f\4> <NS-lI-fNIL \5>}/ && ($j=51)) ||
        # polar question: branch off initial AP-taking auxiliary
        (s/{(QS)([^ ]*?)(-[fghjlp][^ ]*) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<.*)}/{\1\2\3 <\1\2-bAS-lM-f\4> <AS-lI-fNIL \5>}/ && ($j=52)) ||
        # polar question: branch off initial LP-taking auxiliary  ***NOTE: 's AND 'd WON'T GET USED***
        (s/{(QS)([^ ]*?)(-[fghjlp][^ ]*) <([^\]]*VB[A-Z]? (?:[Hh]as|[Hh]ave|[Hh]ad|'s|'ve|'d)[^>]*)> (<.*)}/{\1\2\3 <\1\2-bLS-lM-f\4> <LS-lI-fNIL \5>}/ && ($j=53)) ||
        # polar question: allow subject gap without inversion
        (s/{(QS)([^ ]*-gNS(-[0-9]+)[^ ]*?)(-[fghjlp][^ ]*) <NP[^ ]* \[-NONE- \*T\*\3\]> (<.*)}/{\1\2\4 <VP-fNIL \5>}/ && ($j=54)) ||
        # embedded sentence: delete initial empty interrogative phrase     ****WHY WOULD THIS HAPPEN??? NO WH IN EMBEDDED SENTENCE****
        (s/{(ZS)(?!r)([^ ]*?)(-[fghjlp][^ ]*) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gNS\4-lI-fNIL \5>}/ && ($j=55)) ||

        #### rel clause
        # implicit-pronoun relative: delete initial empty interrogative phrase
        (s/{(ZSr|RC)([^ ]*?)(-[fghjlirp][^ ]*) <WHNP[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gNS\4-lI-fNIL \5>}/ && ($j=56)) ||
        # implicit-pronoun relative: delete initial empty interrogative phrase as adverbial
        (s/{(ZSr|RC)([^ ]*?)(-[fghjlrp][^ ]*) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)}/{\1\2\3 <VS\2-gRP\4-lI-fNIL \5>}/ && ($j=57)) ||
        # branch off initial relative noun phrase
        (s/{(ZSr|RC|RP)([^ ]*?)(?:-rNS)?(-[fghjlp][^ ]*) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2-rNS\3 <NS-rNS-lN-f\4\5\6\7> <VS\2-gNS\6-lI-fNIL \8>}/ && ($j=58)) ||
        # branch off initial relative adverbial phrase with empty subject ('when in rome')
        (s/{(ZSr|RC|RP)([^ ]*?)(?:-rNS)?(-[fghjlp][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (.*\[-NONE- *\*T\*\6\].*)>}/{\1\2-rNS\3 <RP-rNS-lN-f\4\5\6\7> <AP\2-gRP\6-lI-fNIL \8>}/ && ($j=59)) ||
        # branch off initial relative adverbial phrase
        (s/{(ZSr|RC|RP)([^ ]*?)(?:-rNS)?(-[fghjlp][^ ]*) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)}/{\1\2-rNS\3 <RP-rNS-lN-f\4\5\6\7> <VS\2-gRP\6-lI-fNIL \8>}/ && ($j=60)) ||
        # embedded question: branch off initial interrogative RP whether/if
        (s/{(ZSr|RC)([^ ]*?)(?:-rNS)?(-[fghjlp][^ ]*) <(IN[^>]*)> (<.*)}/{\1\2-rNS\3 <RP-iNS-lI-f\4> <VS\2-gRP-lN-fNIL \5>}/ && ($j=61)) ||

        #### middle NS
        # branch off middle modifier AP colon
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<[^ ]* :> <.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL \5>}/ && ($j=62)) ||

        #### conjunction
        # branch final right-node-raising complement NS
        (s/{()([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <NP([^ ]*\5[^>]*)>}/{\1\2\3 <\1\2-hNS\5-lI-fNIL \4> <NS-lA-fNS\6>}/ && ($j=63)) ||
        # branch final right-node-raising modifier AP
        (s/{()([^ ]*?)(-[fghjlp][^ ]*) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <((?:PP)[^ ]*\5[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\6>}/ && ($j=64)) ||
        # pinch ... CC ... -NONE- and re-run
        (s/{([^C][^ ]*?)(-[fghjlp][^ ]*)(?!.*\|) (<.*) (<CC[^>]*>) (<.*) (<[^ ]* \[-NONE- [^\]]*\]>)}/<\1\2 <\1 \3 \4 \5> \6>/ && ($j=65)) ||
        # branch off initial colon in colon...semicolon...semicolon construction
        (s/{([^ ]*)(-[fghjlp][^ ]*)(?!.*\|) (<. :>) <NP([^ ]*)([^>]*)> (<. ;.*<. ;.*)}/{\1\2 \3 <NS\4-lA-fNIL <NS\4\5> \6>}/ && ($j=65.5)) ||
        (s/{([^ ]*)(-[fghjlp][^ ]*)(?!.*\|) (<. :>) <([^ ]*)([^>]*)> (<. ;.*<. ;.*)}/{\1\2 \3 <\4-lA-fNIL <\4\5> \6>}/ && ($j=66)) ||
        # branch off initial conjunct prior to semicolon delimiter
        (s/{([^C][^ ]*?)(-[fghjlp][^ ]*)(?!.*\|) (<.*?) (<[^ ]* ;> .*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <Cs\1-lI-fNIL \4>}/ && ($j=67)) ||
        # branch off initial conjunct prior to comma delimiter
        (s/{(ZSr[^ ]*?)(-[fghjlrp][^ ]*)(?!.*\|) (<.*?) (<[^ ]* ,> .*<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <Cc\ZR-lI-fNIL \4>}/ && ($j=67.5)) ||
        (s/{([^C][^ ]*?)(-[fghjlrp][^ ]*)(?!.*\|) (<.*?) (<[^ ]* ,> .*<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <Cc\1-lI-fNIL \4>}/ && ($j=68)) ||
        # branch off initial conjunct prior to conj delimiter
        (s/{(RP[^ ]*?)(-[fghjlrpi][^ ]*)(?!.*\|) (<.*?) (<CC[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <C\1-lI-fNIL \4>}/ && ($j=68.5)) ||
        (s/{(ZS[^ ]*?)(-rNS)(-[fghjlp][^ ]*)(?!.*\|) (<.*?) (<CC[^>]*> <.*)}/{\1\2\3 <\1-lC-fNIL \4> <C\1-rNS-lI-fNIL \5>}/ && ($j=68.6)) ||
        (s/{([^C][^ ]*?)(-[fghjlrp][^ ]*)(?!.*\|) (<.*?) (<CC[^>]*> <.*)}/{\1\2 <\1-lC-fNIL \3> <C\1-lI-fNIL \4>}/ && ($j=69)) ||
#        # branch off initial conjunct prior to comma/semi/colon/dash between matching constituents
#        s/{([^C][A-Z]S[^ ]*?)(-[fghjlp][^ ]*) <([^- ]*)([^>]*)> (<[^ ]* ,> <\3[^>]*>)}/{\1mod\2 <\1-lC-f\3\4> <C\1-lI-fNIL \5>}/ ||
        # branch off initial semicolon delimiter
        (s/{(Cs)([^- ]*)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*) (;)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;).*)}/{\1\2\3\4 <\5-lM-f\5 \6> <\1\2-pS\3-lI-fNIL \7>}/ && ($j=70)) ||
        # branch off initial comma delimiter
        (s/{(Cc)([^- ]*)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*) (,)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not).*)}/{\1\2\3\4 <\5-lM-f\5 \6> <\1\2-pC\3-lI-fNIL \7>}/ && ($j=71)) ||
        # branch off initial conj delimiter and final conjunct (and don't pass -p down)
        (s/{(C[sc])(ZR)([^ ]*?)(-p[SC])(-[fghjlp][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <ZSr\3-lC-fNIL \7>}/ && ($j=71.5)) ||
        (s/{(C[sc])([^- ]*)([^ ]*?)(-p[SC])(-[fghjlp][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <\2\3-lC-fNIL \7>}/ && ($j=72)) ||
        # branch off initial conj delimiter and final conjunct (no -p to remove)
        (s/{(C[sc]?)(ZR)([^ ]*?)()(-[fghjlp][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <ZSr\3-lC-fNIL \7>}/ && ($j=73)) ||
        (s/{(C[sc]?)([^- ]*)([^ ]*?)()(-[fghjlp][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*)> (<.*)}/{\1\2\3\4\5 <CC-lI-f\6> <\2\3-lC-fNIL \7>}/ && ($j=73)) ||
#        # branch off initial comma/semi/colon/dash between matching constituents
#        s/{(C)([^sc][^- ]*)([^ ]*?)()(-[fghjlp][^ ]*) (<[^ ]* (?:,|;|:|--|-)>) (<.*)}/{\1mod\2\3\4\5 \6 <\2\4-lI-fNIL \7>}/ ||
        # branch off initial conjunct prior to semicolon delimiter
        (s/{(Cs)([^- ]*)([^ ]*?)(-pS)(-[fghjlp][^ ]*) (<.*?) (<[^ ]* ;> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ && ($j=74)) ||
        # branch off initial conjunct prior to comma delimiter
        (s/{(Cc)([^- ]*)([^ ]*?)(-pC)(-[fghjlp][^ ]*) (<.*?) (<[^ ]* ,> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ && ($j=75)) ||
        # branch off initial conjunct prior to conj delimiter (and don't pass -p down)
        (s/{(C[sc])(ZR)([^ ]*?)(-p[SC])(-[fghjlp][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)}/{\1\2\3\4\5 <ZSr\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ && ($j=75.5)) ||
        (s/{(C[sc])([^- ]*)([^ ]*?)(-p[SC])(-[fghjlp][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ && ($j=76)) ||
        # branch off initial conjunct prior to conj delimiter (no -p to remove)
        (s/{(C[sc]?)([^- ]*)([^ ]*?)()(-[fghjlp][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)}/{\1\2\3\4\5 <\2\3-lC-fNIL \6> <\1\2\3-lI-fNIL \7>}/ && ($j=77)) ||

        ######################################################################
        ## 4. LOW PRECEDENCE BRANCHING FINAL CONSTITUENTS

        # branch off final parenthetical sentence with extraction
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <VS-gSS\7-lN-f\5\6>}/ && ($j=78)) ||

        # branch NS -> DD AI: 'the best' construction
        (s/{(NS)([^ ]*?)(-[fghjlp][^ ]*) <(?:DT)([^>]*)> <(?:RB|ADJP)([^>]*)>}/{\1\2\3 <DD-lM-f\4> <AI-lI-f\5>}/ && ($j=79)) ||

        #### final VP|IP|BP|LP|AP (following auxiliary)
        # branch off final VP as argument BP
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*(?:TO |MD | do[\]>]| does[\]>]| did[\]>]).*?) (<RB.*)?(<VP.*>)}/{\1\2\3 <\1\2-bBP-lM-fNIL \4> <BP-lI-fNIL \5\6>}/ && ($j=80)) ||
        # branch off final VP as argument LP (w. special cases b/c 's ambiguous between 'has' and 'is')
        (s/{([VIBLAG]P)([^ ]*?)(-[fghjlp][^ ]*) (.*(?: have| having| has| had| 've|VBD 'd)>.*?) (<RB.*)?(<VP.*>)}/{\1\2\3 <\1\2-bLP-lM-fNIL \4> <LP-lI-fNIL \5\6>}/ && ($j=81)) ||
        (s/{([VIBLAG]P)([^ ]*?)(-[fghjlp][^ ]*) (.*<VBZ *'s>.*?) (<RB.*)?(<VP[^\]]* (?:$LP_FIRST).*>)}/{\1\2\3 <\1\2-bLP-lM-fNIL \4> <LP-lI-fNIL \5\6>}/ && ($j=82)) ||
        # branch off final PRT as argument AP particle
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(PRT)([^ ]*) \[RP ([^ ]*)\]>}/{\1\2\3 <\1\2-bAP\7-lI-fNIL \4> <AP\7-lA-f\5\6 \7>}/ && ($j=83)) ||
        # branch off final modifier RP (extraposed from argument)    **TO PRESERVE EXTRAPOSN: /{\1\2\3 <\1\2-gRP\5-lI-fNIL \4> <RP-lM-f\5\6\7>}/ ||
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<VB.*<.*\[-NONE- \*ICH\*(-[0-9]+)(?![0-9]).*) <(VP[^ ]*)\5((?![0-9])[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\6\5\7>}/ && ($j=84)) ||
        # branch off final VP|ADJP as argument AP
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (.*(?: be| being| been| is| was|VBZ 's| are| were| 're)>.*?) (<RB.*)?(<(?:VP|VB[DNG]|ADJP|JJ|CD|PP[^ ]*-PRD|IN|UCP|ADVP[^ ]*-PRD|SBAR[^ ]*-PRD (?!\[WH|\[IN that)).*>)}/{\1\2\3 <\1\2-bAP-lM-fNIL \4> <AP-lI-fNIL \5\6>}/ && ($j=85)) ||
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (.*(?: be| being| been| is| was|VBZ 's| are| were| 're)>.*?) (<RB.*)?(<(?:NP|NN|S[^ ]*-NOM|SBAR[^ ]*-NOM|SBAR[^ ]*-PRD).*>)}/{\1\2\3 <\1\2-bAP-lM-fNIL \4> <AP-lI-fNIL <NS-lI-fNIL \5\6>>}/ && ($j=86)) ||
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(ADJP|PRT|ADVP[^ ]*-PRD|PP[^ ]*-PRD|VP$NO_VP_HEAD \[VB[NG])([^>]*)>}/{\1\2\3 <\1\2-bAP-lI-fNIL \4> <AP-lA-f\5\6>}/ && ($j=87)) ||
        (s/{(AP|RP)([^ ]*?)(-[fghjlp][^ ]*) (<IN[^>]*>) <(JJ)([^>]*)>}/{\1\2\3 <\1\2-bAP-lI-fNIL \4> <AP-lA-f\5\6>}/ && ($j=88)) ||
        # branch off final argument embedded question SS w. quotations
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBARQ[^>]*)>}/{\1\2\3 <\1\2-bSS-lI-fNIL \4> <SS-lA-f\5>}/ && ($j=89)) ||

        #### final NS
        # delete final empty object of passive
        (s/{(AP|RP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]>}/<\1\2\3 \4>/ && ($j=90)) ||
        # delete final *PPA*- empty category
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <[^ ]* \[-NONE- \*PPA\*-[^ ]*\]>}/<\1\2\3 \4>/ && ($j=91)) ||
        # branch off final IN|TO + NS as modifier RP
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<(?:IN|TO)[^>]*> <NP[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ && ($j=92)) ||
        # branch off final argument OS
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(WHPP[^ ]* \[IN of\][^>]*)>}/{\1\2\3\4 <\1\2-bOS-lI-fNIL \5> <OS\3-lA-f\6>}/ && ($j=93)) ||
        # branch off final argument OS
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(PP[^ ]* \[IN of\][^>]*)>}/{\1\2\3\4 <\1\2\3-bOS-lI-fNIL \5> <OS-lA-f\6>}/ && ($j=94)) ||
        # branch off final argument OS
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) (<IN of> <[^>]*>)}/{\1\2\3\4 <\1\2\3-bOS-lI-fNIL \5> <OS-lA-fNIL \6>}/ && ($j=95)) ||
        # branch off final argument GP
        (s/{([VIBLAGR]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(S(?:[-=][^ ]*)?-NOM(?![^ ]*-TMP)[^ ]*) \[NP-SBJ[^ ]* \[-NONE- [^\]]*\]\]([^>]*)>}/{\1\2\3\4 <\1\2-bGP-lI-fNIL \5> <GP\3-lA-f\6\7>}/ && ($j=96)) ||
        # branch off final argument GS
        (s/{([VIBLAGR]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(S(?:[-=][^ ]*)?-NOM)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bGS-lI-fNIL \5> <GS\3-lA-f\6\7>}/ && ($j=97)) ||
        # branch off final argument NS
		# special handling for "no matter"
        (s/{([R]P)([^ ]*?)(-i[^- ]*)(-[fghjlp][^ ]*) (<DT [Nn]o>) <(NP|DT|NN|WHNP|S[^ ]*-NOM)(?![^ ]*-TMP)([^>]*matter)>}/{\1\2\3\4 <\1\2\3-bNS-lI-fNIL \5> <NS-lA-f\6\7>}/ && ($j=97.4)) ||
        (s/{([RA]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(NN)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bNS-lI-fNIL \5> <NS\3-lA-fNP\7>}/ && ($j=97.45)) ||
        (s/{([LVGB]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(NP)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2\3-bNS-lI-fNIL \5> <NS-lA-fNS\7>}/ && ($j=97.46)) ||
        (s/{([IAR]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(NP)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bNS-lI-fNIL \5> <NS\3-lA-fNS\7>}/ && ($j=97.47)) ||
        (s/{([R]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(DT|WHNP|S[^ ]*-NOM)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bNS-lI-fNIL \5> <NS\3-lA-f\6\7>}/ && ($j=97.5)) ||
        (s/{([VIBLAGR]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(NP|DT|NN|WHNP|S[^ ]*-NOM)(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bNS-lI-fNIL \5> <NS\3-lA-f\6\7>}/ && ($j=98)) ||
        # gerund: branch off final argument NS
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(NP)([^>]*)>}/{\1\2\3 <\1\2-bNS-lI-fNIL \4> <NS-lA-fNS\6>}/ && ($j=98.5)) ||
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(S[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-bNS-lI-fNIL \4> <NS-lA-f\5\6>}/ && ($j=99)) ||

        # branch off final modifier empty S|SBAR expletive trace (keep around for use in cleft)
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <([^>\]]*-NONE- \*EXP\*[^>\[]*)>}/{\1e\2\3 <\1\2-lI-fNIL \4> <\5>}/ && ($j=100)) ||

        #### final S
        # branch off final S-ADV with empty subject as modifier RP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE[I]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL <IP-lI-f\5 \6>>}/ && ($j=101)) ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE[A]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5 \6>}/ && ($j=102)) ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE[I]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL <IP-lI-f\5 \6>>}/ && ($j=103)) ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE[A]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5 \6>}/ && ($j=104)) ||
        # branch off final S with empty subject as argument BP|IP|AP|VP
        (s/{(SS|VS\-iNS|QS|ZS(?!r)|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>}/{\1\2\3 <\1\2-b\6P-lI-fNIL \4> <\6P-lA-f\5 \7>}/ && ($j=105)) ||
        (s/{(SS|VS\-iNS|QS|ZS(?!r)|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([VIBA])P[^ ]*) ([^>]*)>}/{\1\2\3 <\1\2-b\6P-lI-fNIL \4> <\6P-lA-f\5 \7>}/ && ($j=106)) ||
        # branch off final 'so' + S as modifier RP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<[^>\]]* so\]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ && ($j=107)) ||
        # branch off final IN|TO + S as modifier RP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<(?:IN)[^ ]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL \5>}/ && ($j=108)) ||
        # branch off final IN|TO + S as modifier AP
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<(?:IN|TO)[^>]*> <S-TOBE[V]S[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lN-fNIL \5>}/ && ($j=109)) ||
        # branch off final S-ADV as modifier RP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([IA])S[^ ]*(?:-ADV|-PRP)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lN-fNIL <\6S-lI-fS\5>>}/ && ($j=110)) ||
        # branch off final S as argument VS
        (s/{(ZS)(?!r)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([V])S[^>]*)>}/{\1\2\3 <\1\2-b\6S-lM-fNIL \4> <\6S-lI-f\5>}/ && ($j=111)) ||
        # branch off final ADVP + S as modifier VS|IS|BS|AS
        (s/{(SS|VS\-iNS|QS|[VIBLAG]S)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<ADVP[^>]*>) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\7S-lN-fNIL \5 <\7S-lI-f\6>>}/ && ($j=112)) ||
        # branch off final S as modifier VS|IS|BS|AS
        (s/{(SS|VS\-iNS|QS|[VIBLAG]S)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <\6S-lN-f\5>}/ && ($j=113)) ||
        # branch off final S as argument VS|IS|BS|AS
#        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-b\6S-lI-fNIL \4> <\6S-lA-f\5>}/ && ($j=114)) ||
        (s/{(NS)([^ ]*?)(-[fghjlpi][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-b\6S-lI-fNIL \4> <\6S-lA-f\5>}/ && ($j=114)) ||
        (s/{([VIBLAGR]P|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(S-TOBE([VIBA])S[^>]*)>}/{\1\2\3 <\1\2-b\6S-lI-fNIL \4> <\6S-lA-f\5>}/ && ($j=114.5)) ||

        #### final SBAR
        # branch off final SBAR as modifier AP NS (nom clause):                        {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NS <NS ...> <AP <NS WH# ... t# ...>>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) ((?!.*<CC)<.*) <(SBAR[Q]?-NOM)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <NS-lI-f\5\6>>}/ && ($j=115)) ||
        # branch off final SBAR as modifier ZSr:                                        {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NS <NS ...> <ZSr WH# ... t# ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR$SUITABLE_REL_CLAUSE)>}/{\1\2\3 <\1\2-lI-fNIL \4> <ZSr-rNS-lN-f\5>}/ && ($j=116)) ||
        # branch off final SBAR as argument NS (nom clause):                           {VP ... <SBAR [WH# what/who] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/{([VIBLAGR]P)([^ ]*?)(-[ir][^- ]*)?(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]*-NOM|SBAR [^\]]*(?:when|whether))(?![^ ]*-TMP)([^>]*)>}/{\1\2\3\4 <\1\2-bNS-lI-fNIL \5> <NS\3-lA-f\6\7>}/ && ($j=117)) ||
        # branch off final SBAR as argument NS (gerund's nom clause):                  {NS ... <SBAR [WH# what/who] ... t# ...>} => {NS <NS-bNS ...> <NS ... t# ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<(?:(?!VB|JJ|MD|TO|NP)[^>])*VBG[^>]*>) <(SBAR[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-bNS-lI-fNIL \4> <NS-lA-f\5\6>}/ && ($j=118)) ||
        # branch off final SBAR as argument IP:                                        {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ...]>} => {VP-bIP ... <IP ... to ...>}
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/{\1\2\3 <\1\2-bIP-lI-fNIL \4> <IP-lA-fNIL \6>}/ && ($j=119)) ||
#        # branch off final SBAR as modifier RP IP:                                    ----> {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ... t# ...]>} => {VP ... <IP ... to ...>}
#        s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/{\1\2\3 <\1\2-bIP-lI-fNIL \4> <IP-lA-fNIL \6>}/ ||
#        # branch off final SBAR as modifier RP IP:                                    {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ...]>} => {VP ... <RP [IP ... to ...]>}
#        s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/<\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP IP (from SBAR trace coindexed to empty subj)
#        s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>}/<\1\2\3 <\1\2-lI-fNIL \4> <AP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP (from SBAR trace coindexed to final modifier, which must have gap discharged)
#        s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]* \[WHADVP(-[0-9]+) [^>]*) \[ADVP[^ ]* \[-NONE- \*T\*\6\]\]([\]]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-f\5\7>}/ ||
        # branch off final SBAR as argument ES-gNS ('tough for X to Y' construction):  {AP ... <SBAR [WH# nil] ... for ... #t ...>} => {AP <AP-bESg ...> <ES-gNS ... for ... #t ...>}  ****SHOULD BE ES..-lN****
        (s/{(AP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] (\[IN for\] [^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)>}/{\1\2\3 <\1\2-bESg-lI-fNIL \4> <ES-gNS\5-lM-fNIL \6>}/ && ($j=120)) ||
        # branch off final SBAR as argument IP-gNS ('tough to Y' construction):        {AP ... <SBAR [WH# nil] ... to ... #t ...>} => {AP <AP-bIPg ...> <IP-gNS ... for ... #t ...>}
        (s/{(AP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[\VP[^ ]* \[TO to\][^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>}/{\1\2\3 <\1\2-bIPg-lI-fNIL \4> <IP-gNS\5-lA-fNIL \6>}/ && ($j=121)) ||
        # branch off final SBAR as argument AP-gNS ('worth Y-ing' construction):       {AP ... <SBAR [WH# nil] ... #t ...>} => {AP <AP-bAPg ...> <AP-gNS ... #t ...>}
        (s/{(AP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>}/{\1\2\3 <\1\2-bAPg-lI-fNIL \4> <AP-gNS\5-lA-fNIL \6>}/ && ($j=122)) ||
        # branch off final SBAR as modifier IP-gNS (NS_i [for X] to find pics of t_i): {NS ... <SBAR [WHNP# nil] ... #t ...>} => {NS <AP ...> <IP-gNS ... #t ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]*) \[WHNP[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>}/{\1\2\3 <\1\2-lI-fNIL \4> <IP-gNS\6-lN-f\5 \7>}/ && ($j=123)) ||
        # branch off final SBAR as modifier IP-gRP (NS_i [for X] to say you ... t_i):  {NS ... <SBAR [WH# nil] ... #t ...>} => {NS <AP ...> <IP-gRP ... #t ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]*) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>}/{\1\2\3 <\1\2-lI-fNIL \4> <IP-gRP\6-lN-f\5 \7>}/ && ($j=124)) ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# where/when] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/{(SS|VS\-iNS|QS|ZS(?!r)|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-lN-f\5\6>}/ && ($j=125)) ||
        # branch off final SBAR as modifier ZSr:                                        {NS ... <SBAR [WH# where/when] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <ZSr-lN-f\5\6>}/ && ($j=126)) ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# which] ... t# ...>} => {VP <VP ...> <RC WH# ... t# ...>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*which[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-rNS-lN-f\5\6>}/ && ($j=127)) ||
        # branch off final SBAR as modifier ZSr (that|nil):                             {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {VP <VP ...> <ZSr WH# ... t# ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <ZSr-rNS-lN-f\5\6>}/ && ($j=128)) ||
        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-bQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR(?!-ADV|-TMP)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that|[^\]]*-NONE-))([^>]*)>}/{\1\2\3 <\1\2-bVS-iNS-lI-fNIL \4> <VS-iNS-lA-f\5\6>}/ && ($j=129)) ||
#        s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+(?![^\]]*what|[^\]]*why)[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RC-lN-f\5\6>}/ ||
        # branch off final SBAR as modifier RP colon:                                  {VP ... <:> <SBAR [IN because/..] ...>} => {VP <VP ...> <AP <:> <AP because ...>>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*<SBAR[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ && ($j=130)) ||
        # branch off final SBAR as modifier AP colon:                                  {NS ... <:> <SBAR [IN because/..] ...>} => {NS <NS ...> <AP <:> <AP because ...>>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<[^ ]* :> <.*<SBAR[^>]*)}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL \5>}/ && ($j=131)) ||
        # branch off final SBAR as modifier ES:                                        {NS ... <SBAR [IN because/..] ...>} => {NS <NS ...> <AP ...>}
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]* \[[^ ]* for\] [^>]*)>}/{\1\2\3 <\1\2-bES-lI-fNIL \4> <ES-lA-f\5>}/ && ($j=132)) ||
        # branch off final SBAR as modifier RP:                                        {VP ... <SBAR [IN because/..] ...>} => {VP <VP ...> <AP ...>}
#WORSE:        s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*-TMP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-t-lM-f\5\6>}/ ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*(?:-ADV|-LOC|-TMP|-CLR)|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5\6>}/ && ($j=133)) ||
        # branch off final SBAR as modifier AP:                                        {NS ... <SBAR [IN because/..] ...>} => {NS <NS ...> <AP ...>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[Q]?-LOC|SBAR[Q]?-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ && ($j=134)) ||
        # branch off final SBAR as argument AP NS (nom clause following 'being'):      {NS ... being ... <SBAR [WH# what/who] ... t# ...>} => {NS <NS-bAP ... being ...> <AP <NS ... t# ...>>}
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (.* being>.*) <(SBAR-NOM|SBAR-PRD)([^>]*)>}/{\1\2\3 <\1\2-bAP-lM-fNIL \4> <AP-lI-fNIL <NS-lI-f\5\6>>}/ && ($j=135)) ||
#        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-bQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
#        s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)>}/{\1\2\3 <\1\2-bVS-iNS-lI-fNIL \4> <VS-iNS-lA-f\5\6>}/ ||
        # delete final empty SBAR given gap ZSr and re-run:
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP]|NS|NP)(-gZR)(-[0-9]+)([^ ]*) (<.*) <SBAR[^>\]]*\[-NONE- \*(?:T|ICH)\*\3\][^>\[]*>}/<\1\2\3\4 \5>/ && ($j=136)) ||
        # branch off final SBAR as argument ZS:                                        {XP ... <SBAR [IN that/nil] ...>} => {XP <XP-bZS ...> <ZS ...>}   *****SHOULD BE ZS******
        (s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SBAR)([^>]*)>}/{\1\2\3 <\1\2-bZS-lI-fNIL \4> <ZS-lA-f\5\6>}/ && ($j=137)) ||

        #### final RP
        # branch off final modifier RP colon
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ && ($j=138)) ||
        # branch off final modifier RP
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?![^ ]*-PRD)(NP-TMP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNS-TMP\6>}/ && ($j=138.5)) ||
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAGR][SP])([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?![^ ]*-PRD)(RB|ADVP|PP|UCP|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-f\5\6>}/ && ($j=139)) ||
        # branch off final SQ|SINV as argument SS
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SQ|SINV)([^>]*)>}/{\1\2\3 <\1\2-bSS-lI-fNIL \4> <SS-lA-f\5\6>}/ && ($j=140)) ||
        # branch off final INJP as argument ZS
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(INTJ)((?![^ ]*-ADV)[^>]*)>}/{\1\2\3 <\1\2-bZS-lI-fNIL \4> <ZS-lA-f\5\6>}/ && ($j=141)) ||
        # branch off final argument LS ('had I only known' construction)
        (s/{([VIBLAGR]P)([^ ]*?)(-[fghjlp][^ ]*) (<VBD had>) (.*<NP.*<VP.*)}/{\1\2\3 <\1\2-bLS-lM-fNIL \4> <LS-lI-fNIL \5>}/ && ($j=142)) ||

        #### final AP
#        # semi/dash splice
#        s/{(NS)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<[^ ]* ;>) (<.*)}/{\1\2\3 <\1\2-lI-fNIL \4> <C\1 \5 <\1\2-lI-fNIL \6>>}/ ||
        # gerund: delete initial empty subject (passing -o to head)
        (s/{(NS)([^ ]*?)(-[fghjlp][^ ]*) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)}/<\1\2\3 \4>/ && ($j=143)) ||
        # branch off final modifier AP appositive NS (passing -o to head)
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) ((?=.*<(?:NP|[^ ]*-NOM).*<(?:N[PS]|[^ ]*-NOM))(?!.*<CC)<.*) <(NP|[^ ]*-NOM)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <NS-lI-fNS\6>>}/ && ($j=144)) ||

        # branch off final modifier AP infinitive phrase (with TO before any VBs) (passing -o to head)  ****************
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(VP(?:(?!\[VB)[^>])*\[TO[^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-fNIL <IP-lI-f\5>>}/ && ($j=145)) ||
        # branch off final modifier AP (passing -o to head)
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(RRC|PP|ADJP|ADVP|RB|UCP|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ && ($j=146)) ||
        # branch off final modifier AP (from VP) (passing -o to head)
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (.*<(?:NP|NN).*) <(VP)([^>]*)>}/{\1\2\3 <\1\2-lI-fNIL \4> <AP-lM-f\5\6>}/ && ($j=147)) ||
        # branch off final argument LP (passing -o to head)
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (.* having>.*) <(VP)([^>]*)>}/{\1\2\3 <\1\2-bLP-lM-fNIL \4> <LP-lI-f\5\6>}/ && ($j=148)) ||
        # branch off final argument VS\-iNS (passing -o to head)  **WH VS\-iNS WILL NEVER FIRE**
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(SQ(?=[^ ]* \[WH))([^>]*)>}/{\1\2\3 <\1\2-bVS-iNS-lI-fNIL \4> <VS-iNS-lA-f\5\6>}/ && ($j=149)) ||

        #### final misc needed by SS   *****COULD BE GENERALIZED******
        # branch off final 'so' + S
        (s/{(SS|VS\-iNS|QS|ZS|ES|[VIBLAG]S)([^ ]*?)(-[fghjlp][^ ]*) (<.*) (<[^ ]* so> <S(?![A-Z])[^>]*>)}/{\1\2\3 <\1\2-lI-fNIL \4> <RP-lM-fNIL \5>}/ && ($j=150)) ||

        ######################################################################
        ## 5. LOW PRECEDENCE BRANCHING INITIAL CONSTITUENTS

        #### VS
        # inverted declarative sentence: branch off final subject
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) (<.*) <NP([^ ]*-SBJ)([^>]*)>}/{\1S\2\3 <\1P\2-lI-fNIL \4> <NS-lA-fNS\5\6>}/ && ($j=151)) ||
        # [VIBLAG] sentence: branch off initial ZS subject
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) <(SBAR[^ ]*-SBJ[^\]]*that)([^>]*)> (<.*)}/{\1S\2\3 <ZS-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ && ($j=152)) ||
#        # [VIBLAG] sentence: branch off initial VS\-iNS subject
#        s/{([VIBLA])S([^ ]*?)(-[fghjlp][^ ]*) <(SBAR(?!-ADV)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)> (<.*)}/{\1S\2\3 <VS-iNS-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ ||
        # [VIBLAG] sentence: branch off initial ES subject
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) <(SBAR[^ ]*-SBJ[^\]]*for)([^>]*)> (<.*)}/{\1S\2\3 <ES-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ && ($j=153)) ||
        # [VIBLAG] sentence: branch off initial IP subject
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) <(SheadI[^ ]*-SBJ)([^>]*)> (<.*)}/{\1S\2\3 <IP-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ && ($j=154)) ||
        # [VIBLAG] sentence: branch off initial NS subject
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) <(NP)(-SBJ)?([^>]*)> (<.*)}/{\1S\2\3 <NS-lA-fNS\5\6> <\1P\2-lI-fNIL \7>}/ && ($j=154.5)) ||
        (s/{([VIBLAG])S([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*-NOM|[^ ]*-SBJ)([^>]*)> (<.*)}/{\1S\2\3 <NS-lA-f\4\5> <\1P\2-lI-fNIL \6>}/ && ($j=155)) ||

        #### NS
        # branch off initial punctuation
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]*) ($INIT_PUNCT)> (<.*)}/{\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>}/ && ($j=156)) ||
        # branch off initial parenthetical sentence with extraction
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)}/{\1\2\3 <VS-gSS\6-lN-f\4\5> <\1\2-lI-fNIL \7>}/ && ($j=157)) ||
        # branch off initial pre-determiner
        (s/{(NS)([^ ]*?)(-[ir][^- ]*)?([^ ]*?)(-[fghjlp][^ ]*) <(PDT|RB)([^>]*)> (<.*)}/{\1\2\3\4\5 <DD-lM-f\6\7> <NS\2\3\4-lI-fNIL \8>}/ && ($j=158)) ||
        # branch off initial determiner (wh)
        (s/{(NS)([^ ]*?)(-[ir][^- ]*)?([^ ]*?)(-[fghjlp][^ ]*) <(WDT|WP\$)([^>]*)> (<.*)}/{\1\2\3\4\5 <DD\3-lM-f\6\7> <NP\2\4-lI-fNIL \8>}/ && ($j=159)) ||
        # branch off initial determiner (non-wh)
        (s/{(NS)([^ ]*?)(-[ir][^- ]*)?([^ ]*?)(-[fghjlp][^ ]*) <NP([^>]*\[POS 's?\])([^>]*)> (<.*)}/{\1\2\3\4\5 <DD-lM-fNS\6\7> <NP\2\3\4-lI-fNIL \8>}/ && ($j=159.5)) ||
        (s/{(NS)([^ ]*?)(-[ir][^- ]*)?([^ ]*?)(-[fghjlp][^ ]*) <(DT|PRP\$|PRP|NP[^>]*\[POS 's?\])([^>]*)> (<.*)}/{\1\2\3\4\5 <DD-lM-f\6\7> <NP\2\3\4-lI-fNIL \8>}/ && ($j=160)) ||
        # branch off initial modifier AI
        (s/{(NS|NP)([^ ]*?)(-[ir][^- ]*)?([^ ]*?)(-[fghjlp][^ ]*) <(WHADJP|WRB)([^>]*)> (.*<(?:DT|NP|NX|NN|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3\4\5 <AI\3-lM-f\6\7> <NP\2\4-lI-fNIL \8>}/ && ($j=161)) ||
        # branch off initial modifier AI
        (s/{(NS|NP)([^ ]*?)(-[fghjlrp][^ ]*?)NP <(NN)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3NS <AI-lM-fNP\5> <NP\2-lI-fNIL \6>}/ && ($j=161.4)) ||
        (s/{(NS|NP)([^ ]*?)(-[fghjlrp][^ ]*) <(NN)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3 <AI-lM-fNP\5> <NP\2-lI-fNIL \6>}/ && ($j=161.5)) ||
        (s/{(NS|NP)([^ ]*?)(-[fghjlrp][^ ]*) <(CD|QP|JJ|ADJP|WHADJP|IN|PP|RB|TO|ADVP|VB|UCP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3 <AI-lM-f\4\5> <NP\2-lI-fNIL \6>}/ && ($j=162)) ||
        (s/{(NS|NP)([^ ]*?)(-[fghjlp][^ ]*) <(NAC)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)}/{\1\2\3 <AI-lM-fNIL <NS-lI-f\4\5>> <NP\2-lI-fNIL \6>}/ && ($j=163)) ||
        # rebinarize QP containing dollar sign followed by *U*, and continue
        (s/{(NS|NP|AI)([^ ]*?)(-[fghjlp][^ ]*) <QP ([^>]*)\[([^ ]* [^ ]*[\$\#][^ ]*)\] ([^>]*)>(?: <-NONE- \*U\*>)?}/<\1\2\3 \4\[\5\] \[QP \6\]>/ && ($j=164)) ||
        # branch off currency unit followed by non-final *U*
        (s/{(NS|NP|AI)([^ ]*?)(-[fghjlp][^ ]*) (<[^ ]* [\$\#][^ ]*>.*) <-NONE- \*U\*> (<.*)}/{\1\2\3 <AI-lM-fNIL \4> <NP-lI-fNIL \5>}/ && ($j=165)) ||
        # rebinarize currency unit followed by QP
        (s/{(NS|NP|AI)([^ ]*?)(-[fghjlp][^ ]*) <([^ ]* [\$\#][^ ]*)> (.*?)( <-NONE- \*U\*>)?}/{\1\2\3 <\$-lI-f\4> <AI-lM-fNIL \5>}/ && ($j=166)) ||

        #### AI|RI|NP
        # branch off initial modifier RI
        (s/{(AI)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <RI\3-lM-f\5> <AI-lI-fNIL \6>}/ && ($j=167)) ||
        (s/{(AI)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <NN([^>]*)> (<.*)}/{\1\2\3\4 <RI-lM-fNP\5> <AI\3-lI-fNIL \6>}/ && ($j=167.5)) ||
        (s/{(AI)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <RI-lM-f\5> <AI\3-lI-fNIL \6>}/ && ($j=168)) ||
#        s/{(AI)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <RI-lM-f\3> <AI-lI-fNIL \4>}/ ||
        (s/{(RI)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <RI\3-lM-f\5> <RI-lI-fNIL \6>}/ && ($j=169)) ||
        (s/{(RI)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <RI-lM-f\5> <RI\3-lI-fNIL \6>}/ && ($j=170)) ||
#        s/{(RI)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <RI-lM-f\3> <RI-lI-fNIL \4>}/ ||
        (s/{(NP)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <(W[^>]*)> (<.*)}/{\1\2\3\4 <AI\3-lM-f\5> <NP-lI-fNIL \6>}/ && ($j=171)) ||
        (s/{(NP)([^ ]*?)(-[ir][^- ]*)?([^ ]*) <([^>]*)> (<.*)}/{\1\2\3\4 <AI-lM-f\5> <NP\3-lI-fNIL \6>}/ && ($j=172)) ||
#        s/{(NP)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <AI-lM-f\3> <NP-lI-fNIL \4>}/ ||
        (s/{(CC)([^ ]*) <([^>]*)> (<.*)}/{\1\2 <CC-lM-f\3> <CC-lI-fNIL \4>}/ && ($j=173)) ||

		######################################################################
        ## 0. TURN {NS..<NP..>} TO {NS..<NS..>}     AND      {NS.. <DT..> <NN..>} TO {NS.. <DT..> <NP..>}  (this is to fix sent 2921)
#        (s/{NS([^ ]*) <NP([^>]*)>([ ]*)}/{NS\1 <NS\2>\3}/ && ($j=0.5)) ||
#        (s/{NS([^ ]*) <DT([^>]*)> <NN([^>]*)>([ ]*)}/{NS\1 <DT\2> <NP\3>\4}/ && ($j=0.6)) ||
        ## {DD.. <NP..> ..} TO {DD.. <NS..> ..}  (this is to fix sent 7419)
#        (s/{DD([^ ]*) <NP(.*)}/{DD\1 <NS\2}/ && ($j=0.7)) ||
#		(s/<NP /<NS /g && ($j=0.1)) ||
#		(s/<NN /<NP /g && ($j=0.2)) ||
        
        #### panic as MWE
#        s/{([VIBLAGR]P|NS|NP)([^ ]*?)(-[fghjlp][^ ]*) (<.*) <(?!-NONE-)([^ ]*) (..?.?.?)>}/{\1\2\3 <\1\2-bAP\6-fNIL \4> <AP\6-lA-f\5 \6>}/ ||

        1 ) {
           	if ($j==0) {
           		while( s/{(.*)<NP(.*)}/{\1<NS\2}/ && ($j=0.1) ) {}  #(this is to fix sent 12157)
        		while( s/{(.*)<NN(P)? (.*)}/{\1<NP\2 \3}/ && ($j=0.2) ) {}  #(this is to fix sent 7919)
        	}
        	debug($step, " used rule $j")
        }

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
           s/{([^ ]*-iNS)(-g[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjlpi][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)}/{\1\2\3\4 <\5\2\3\6>\7}/ ||
           s/{([^ ]*)(-g[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjlp][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)}/{\1\2\3\4 <\5\2\3\6>\7}/ ||
           # propagate -h right-node-raising gap tag from parent to each child dominating trace with matching index
           s/{([^ ]*)(-h[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjlp][^>]*\[-NONE- *\*(?:RNR)\*\3\][^>]*)>(.*)}/{\1\2\3\4 <\5\2\3\6>\7}/ ||
           # add -j it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-j)([^ ]*?)(-[fghjlp][^>]*\[[^ ]*\2[^>]*)>/\1<\3-jNSe\4>/ ||
           # add -a spec tag to sibling of cleft complement following expletive trace (-a then passed down thru syn heads)
           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-a)([^ ]*?)(-[bfghjlp][^>]*)> <([^ ]*\2[^>]*)>/\1<\3-aNSe\4> <\5>/ ||
           # add -a spec tag to sibling of cleft complement containing expletive trace (-a then passed down thru syn heads)
           s/<(?![^ ]*-a)([^ ]*?)(-[bfghjlp][^>]*-NONE- \*EXP\*(-[0-9]+)[^0-9][^>]*)> <([^ ]*\3[^>]*)>/<\1-aNSe\2> <\4>/ ||
#           # turn last -i tag into -a
#           s/{([^ ]*)-i(?!.*<[^ ]*-i.*\})(.*)}/{\1-a\2}/ ||
##           # add -i it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)\{(?![^ ]*-i)([^ ]*?)(-[fghjlp].*\[[^ ]*\2.*)\}/\1\{\3-iNSe\4\}/ ||
##           # turn last -i tag into -a
##           # add -a spec tag to each constituent following expletive trace that immediately dominates cleft complement with matching index (-a then passed down thru syn heads)
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-a)([^ ]*?)(-[fghjlp].*\[[^ ]*\2.*)>/\1<\3-aNSe\4>/ ||
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
  s/-h([A-Z]+)-[0-9]+/-b\1/g;
  # turn expletive it-cleft traces into specifier tags
  s/-j([A-Z]+)/-a\1/g;

  # correct VBNs
  s/\(LP([^ ]*)-fVB[A-Z]*/\(LP\1-fVBN/g;
  s/\((AP|AI|RP|RI)([^ ]*)-fVB[ND]/\(\1\2-v-fVBN-v/g;     ## mark passive
  s/\((AP|AI|RP|RI|NS|NP)([^ ]*)-fVB[G]/\(\1\2-r-fVBG-r/g;   ## mark progressive/gerund

#  s/\(([^ ]*)(-fVB)/\(\1-wasvb\2/g;  ## helps mecommon?

#  # throw out new categories
#  s/[^ ]+-f//g;
#  s/\(VBN-v ([^\)]*)\) *\(NS[^ ]* *\(-NONE- [^\)]*\) *\)/\(VBN-v \1\)/g;
#  s/\(VP \(VBN-v ([^\)]*)\) *\)/\(VP-v \(VBN-v \1\)\)/g;

#  # throw out old categories
#  s/\(([^ ]*)-f[^ ]*/\(\1/g;

# put the synrole -l at the end
#  s/-l([A-Z])([^ ]*)/\2-l\nn1/g;

  # clean up -TOBE..
  s/-TOBE..//g;

  # add unary branch to generalize gap categories
  s/\(([^ ]*)(-l[^I])([^ ]*) ([^\(\)]*)\)/\(\1\2\3 \(\1-lI\3 \4\)\)/g;
  s/\(([^ ]*)-g(NS|SS)([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1-u\2\3 \4\)\)/g;
  s/\(([^ ]*)-g(RP|ZR)([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1\3 \4\)\)/g;

  # change ZSr (previously known as AC back to just ZS)
  s/ZSr/ZS/g;

  # output
  print $_;
}
