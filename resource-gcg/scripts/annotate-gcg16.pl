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


#### CODE REVIEW: get rid of NS!!!!
$NOMINALS = '(?:NP(?![^ ]*-TMP)|NS|NN|QP|S[^ ]*-NOM|SBAR[^ ]*-NOM|SBAR[^ ]*-PRD(?= [<\[]WH| [<\[]IN that))';


#### NOTE!
#s/-[mnp]Q//g;

## for each tree...
while ( <> ) {

  $line++;  if ($line % 1000 == 0) { print stderr "$line lines processed...\n"; }
  
  #### category normalization
  # root category
  s/^\((?!NP|FRAG)/\(S-lS-f/;
  s/^\(FRAG/\(A-aN-lS-fFRAG/;
  s/^\(NP(?!-f)/\(N-lS-fNP/;
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
  $CODA = '(?:-[fghvl0-9][^ \{\}]*|-[fghvl0-9]\{[^ \{\}]*\})+';     ### '-[fghl](?:[^ {}\^]|{[^ {}\^]*})*'; # stuff that arguments get to cut in front of
  $CONJCODA = '(?:-[fghjlp0-9][^ \{\}]*|-[fghjlp0-9]\{[^ \{\}]*\})+';
  $RCONJCODA = '(?:-[fghjlr0-9][^ \{\}]*|-[fghjlr0-9]\{[^ \{\}]*\})+';
  $VARCONJCODA = '(?:-[ghjk0-9][^ \{\}]*|-[ghjk0-9]\{[^ \{\}]*\})+';

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/\^\1\^/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    #debug($step++, "   $_");
debug("o-rule ", "$_");

    # turn conj of null-subj-S's with null subj into null-subj-S with conj of VP's
    s/\^S([^ ]*) <S(?:[-=][^ ]*)? (\[NP[^ ]* \[-NONE- [^\]]*\]\]) ([^>]*)> (.*<CC[^>]*>.*) <S(?:[-=][^ ]*)? \2 ([^>]*)>\^/\^S\1 \2 <VP \3 \4 \5>\^/  && debug("o=1 ", "$_");
    # turn mod of null-subj-S into null-subj-S with mod of VP
    s/\^S(?!(?:-f)?[A-Z])([^ ]*) (<.*) (<NP[^ ]* \[-NONE- [^\]]*\]>) (<VP.*)\^/\^S\1 \3 \2 \4\^/ && debug("o=2 ", "$_");

    #### merge subj of obj ctrl into S  # (NP-1 (NNP Edison) ) (S  (NP-SBJ (-NONE- *-1) )
    s/\^(.*) <NP(-[0-9]+) (.*)> <S([^ ]*) \[NP-SBJ \[-NONE- \*\2\]\] (.*)>\^/\^\1 \[S\4 \[NP-SBJ\2 \3\] \5\]\^/ && debug("o=3 ", "$_");

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    s/\^(.*)\^/<\1>/;
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
    s/\(([^\(\)]*)\)/\^\1\^/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
debug("p-rule ", "$_");

    #### merge subj of obj ctrl into S  # (NP-1 (NNP Edison) ) (S  (NP-SBJ (-NONE- *-1) )
    (s/\^(.*) <NP(-[0-9]+) (.*)> <S(-[^ ]*) \[NP-SBJ \[-NONE- \*\2\]\] (.*)>\^/<\1 \[S\4 \[NP-SBJ\2 \3\] \5\]>/ && ($j=0.5)) ||

    # propagate V up to VP node to determine VP/IP/BP/LP/AP...
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<(?:VB[ZDP]|MD).*)\^/\^VP-TOBEVP\1 \2\^/ && debug("p=1 ", "$_");
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<TO.*)\^/\^VP-TOBEIP\1 \2\^/ && debug("p=2 ", "$_");
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<VB .*)\^/\^VP-TOBEBP\1 \2\^/ && debug("p=3 ", "$_");
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?:(?!<VB|<TO).)*<VB[GN].*)\^/\^VP-TOBEAP\1 \2\^/ && debug("p=4 ", "$_");
    # propagate TOBE.P VP up through VP conj...
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*<VP[^ ]*-TOBE([VIBLA])P.*<CC.*)\^/\^VP-TOBE\3P\1 \2\^/ && debug("p=5 ", "$_");
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*<CC.*<VP[^ ]*-TOBE([VIBLA])P.*)\^/\^VP-TOBE\3P\1 \2\^/ && debug("p=6 ", "$_");
    # propagate first TOBE.P VP up through VP mod...
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? (.*?<VP[^ ]*-TOBE([VIBLA])P.*)\^/\^VP-TOBE\3P\1 \2\^/ && debug("p=7 ", "$_");
    # propagate empty VP as TOBEVP
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? (<-NONE-[^>]*>)\^/\^VP-TOBEVP\1 \2\^/ && debug("p=8 ", "$_");
    # propagate random VP as TOBEAP
    s/\^VP(?![^ ]*-TOBE)([-=][^ ]*)? ((?!.*<VP.*\^).*)\^/\^VP-TOBEAP\1 \2\^/ && debug("p=9 ", "$_");
    # propagate PRD XP as TOBEAS
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? ((?:(?!<VP).)*.*<[^ ]*-PRD.*)\^/\^S-TOBEAS\1 \2\^/ && debug("p=10 ", "$_");
#debug($step, ":( $_") ||
    # propagate TOBE.P VP with empty NP up through S...
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-=]*)? ((?:(?!<VP).)*<NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]> (?:(?!<VP).)*<VP[^ ]*-TOBE([VIBLA])P.*)\^/\^S-TOBE\3P\1 \2\^/ && debug("p=11 ", "$_");
    #s/\^S(?![^ ]*-TOBE|[^ ]*-NOM)([-=][^ ]*)? ((?:(?!<VP).)*)<NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]> ((?:(?!<VP).)*<VP[^ ]*-TOBE([VIBLA])P.*)\^/\^S-TOBE\4P\1 \2\3\^/;
    # propagate TOBE.P VP up through S...
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? ((?:(?!<VP).)*.*<VP[^ ]*-TOBE([VIBLA])P.*)\^/\^S-TOBE\3S\1 \2\^/ && debug("p=12 ", "$_");
    # propagate TOBE.S S up through S conj...
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? (.*<S[^ ]*-TOBE(..).*<CC.*)\^/\^S-TOBE\3\1 \2\^/ && debug("p=13 ", "$_");
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? (.*<CC.*<S[^ ]*-TOBE(..).*)\^/\^S-TOBE\3\1 \2\^/ && debug("p=14 ", "$_");
    # propagate first TOBE.S S up through S mod...
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? (.*?<S[^ ]*-TOBE(..).*)\^/\^S-TOBE\3\1 \2\^/ && debug("p=15 ", "$_");
    # propagate empty S as TOBEVS
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? (<-NONE-[^>]*>)\^/\^S-TOBEVS\1 \2\^/ && debug("p=16 ", "$_");
    # propagate random S as TOBEAS
    s/\^S(?![^ ]*-TOBE)([-=][A-Z0-9\-]*)? ((?!.*<VP.*\^).*)\^/\^S-TOBEAS\1 \2\^/ && debug("p=17 ", "$_");

    ####################
    ## convert inner angles (if any) to bracks...
    while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
    ## convert outer braces to angles...
    s/\^(.*)\^/<\1>/;
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


    ######################################################################
    ## A. COLLAPSE ALL UNARY CONSTITUENTS IN ORIGINAL TREEBANK, KEEPING HIGHEST CAT LABEL

#    while (
#           debug($step, "-- $_") ||
#           #### collapse unary constituents using upper cat, lower -f
#           s/\(([^ ]*-f)[^ ]* \{(?![^ ]*-f)(?![^>\]]*-NONE-[^<\[]*\})(.*)\}\)/{\1\2\^/ ||
#           0 ) {}
    while (
           debug($step, "-- $_") ||
           #### collapse unary constituents using upper cat, lower -f
#           s/\^([^ ]*-f)[^ ]* <NP(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>\^/<\1NS\2>/ ||
#           s/\^([^ ]*-f)[^ ]* <NN(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^>]*)>\^/<\1NP\2>/ ||
           s/\^([^ ]*-f)[^ ]*?(-[^ ]*)? <(?![^ ]*-f)(?![^>\]]*-NONE-[^>\[]*>)([^ ]*) ([^>]*)>\^/<\1\3\2 \4>/ ||
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
##        s/\^(.*) \|S([^ ]*-NOM[^\|]*)\|(.*)\^/<\1 <SheadN\2>\3>/ ||
###        s/\^(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<VP$NO_VP_HEAD \[TO [^\|]*)\|(.*)\^/<\1 <SheadInosubj\2>\3>/ ||
###        s/\^(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<(?:VP|ADJP)$NO_VP_HEAD \[(?:VB[NG]|JJ[RS]*) [^\|]*)\|(.*)\^/<\1 <SheadAnosubj\2>\3>/ ||
###        s/\^(.*) \|S([^\|]* ${EMPTY_SUBJ} [^\|]*<[^ ]*-PRD[^\|]*)\|(.*)\^/<\1 <SheadAnosubj\2>\3>/ ||
#        s/\^(.*) \|S([^\|]*<VP$NO_VP_HEAD \[VB [^\|]*)\|(.*)\^/<\1 <SheadB\2>\3>/ ||
#        s/\^(.*) \|S([^\|]*<VP$NO_VP_HEAD \[TO [^\|]*)\|(.*)\^/<\1 <SheadI\2>\3>/ ||
#        s/\^(.*) \|S([^\|]*<(?:VP|ADJP)$NO_VP_HEAD \[(?:VB[NG]|JJ[RS]*) [^\|]*)\|(.*)\^/<\1 <SheadA\2>\3>/ ||
#        s/\^(.*) \|S([^\|]*<[^ ]*-PRD[^\|]*)\|(.*)\^/<\1 <SheadA\2>\3>/ ||
#        s/\^(.*) \|S([^\|]*)\|(.*)\^/<\1 <SheadV\2>\3>/ ||
#        # create transparent constituents
#        s/\^(.*) <S(?!-NOM)(?!head)(?![A-Z])([^>]*)>(.*)\^/<\1 \|S\2\|\3>/ ||

        ######################################################################
        ## 1. UNARY REANNOTATIONS

        #### remove empty NP from S turned to [VIBLAR]P
        (s/\^([VIBLAR]-aN(?!-x)[^ ]*.*) <NP[^ ]*-SBJ[^ ]* \[-NONE- \*[^A-Z ]*\]>(.*)\^/<\1\2>/ && ($j=1)) ||

        #### it-cleft
        (s/\^([SQCFV](?=[^ ]*-fS[^ ]*-CLF)[^ ]*)($CODA) (.*)<NP-SBJ([^>]*)>(.*)\^/<\1\2 \3<NP-CLF\4>\5>/ && ($j=1.5)) ||

        #### unary expand anything to U if all terms are NNP|NNPS
        (s/\^([^U][^ ]*?)(-f[^ ]*-TTL[^ ]*|-f[^ ]*-HLN[^ ]*) (.*)\^/\^\1\2 <U-lI\2 \3>\^/ && ($j=1.6)) ||
        (s/\^([^U][^ ]*?)($CODA) ((?:[\[<]*NNPS?[^ ]* [^\[<]*)*)\^/\^\1\2 <U-lI-fNIL \3>\^/ && ($j=1.7)) ||
        (s/\^([^U][^ ]*?)(-fNNPS?[^ ]*) ([^\[<]*)\^/\^\1\2 <U-lI\2 \3>\^/ && ($j=1.8)) ||

        #### parentheticals
        # identify ptb tag dominating its own trace, and change it to *INTERNAL*
        (s/\^([^ ]*)(-f[^ ]*)(-[0-9]+)((?![0-9]).*-NONE- \*)(?:T|ICH)(\*\3(?![0-9]).*)\^/<\1\2\3\4INTERNAL\5>/ && ($j=2)) ||
        # flatten PRN nodes
        (s/\^(.*) <PRN([^>]*)>(.*)\^/<\1\2\3>/ && ($j=3)) ||
        # subj aux inversion (Q rule): Q-b(X-aN)-bN -> V-aN-b(X-aN)
        (s/\^([QV]-[ab])(\{.-aN\})-bN([^ ]*)($CODA) ([^ ]*)\^/\^\1\2-bN\3 <V-aN-b\2-lI-fNIL \5>\^/ && ($j=3.5)) ||

        #### unary expand N...-fNNS? to N-aD... with elision of determiner
        (s/\^(?=[^ ]*-fNNS?[^ ]* )(?!N-aD)(N|A-aN-x|A-aN|R-aN-x|R-aN)([^ ]*?)($CODA) ([^<\[]*)\^/\^\1\2\3 <N-aD\2-lI-fNIL \4>\^/ && ($j=13.5)) ||

        #### predicative NPs
        # unary expand to NS
#        (s/\^([AR]-aN(?!-x))([^ ]*?)(-g[^ ]*)?-fN[SP]([^ ]*) (.*)\^/\^\1\2\3-fNS\4 <N\2-lI-fNIL \5>\^/ && ($j=12)) ||
        (s/\^([AR]-aN(?!-x))([^ ]*?)(?=[^ ]*-f$NOMINALS)($CODA) (.*)\^/\^\1\2\3 <N\2-lI-fNIL \4>\^/ && ($j=12)) ||
        # unary expand to NS (nom clause)
        (s/\^(A-aN(?!-x))([^ ]*?)(-g[^ ]*)?(-fSBAR[^ ]*(?= <WH)) (<.*)\^/\^\1\2\3\4 <N\2-lI-fNIL \5>\^/ && ($j=13)) ||

        ######################################################################
        ## 2. HIGH BRANCHING / LOW PRECEDENCE FINAL CONSTITUENT

        #### SS
#        # attach final adverbial as high as possible
#        (s/\^([V][^ ]*?)($CODA) (<.*) <((?:ADVP|PP)(?![^ ]*-PRD)[^ ]*)>\^/\^\1\2 <\1-lI-fNIL \3> <R-aN-lM-f\4>\^/ && ($j=3.7)) ||
        # attach final extracted adverbial as high as possible: V-aN-g{R-aN}-1-lI-fVP-TOBEVP <VBZ 's> <NP-PRD ...> <ADVP-TMP [-NONE- *T*-1]>
        (s/\^([SQCFVIBLAGR][^ ]*?)(-g\{R-aN})(-[0-9]+)($CODA) (<.*) <(ADVP(?![^ ]*-PRD)[^ ]* \[-NONE- \*T\*\3\])>\^/\^\1\2\3\4 <\1-lI-fNIL \5> <R-aN-lM-f\6>\^/ && ($j=3.7)) ||
        # semi/dash splice between matching constituents
        (s/\^(S(?!-[cp]))([^ ]*?)($CODA) <([^- ]*)([^>]*)> <([^ ]*) (;|--)> <\4([^>]*)>\^/\^\1\2\3 <\1\2-lC-f\4\5> <\1-pPs-c\1-lI-fNIL <\6-lI-f\6 \7> <\1\2-lC-f\4\8>>\^/ && ($j=4)) ||
#        # inverted sentence: branch off final raised complement SS (possibly quoted) with colon
#        s/\^(SS)([^ ]*?)($CODA) (<.*\[-NONE- *\*ICH\*(-[0-9]+)(?![0-9]).*) <: :> <(S)([^ ]*)\5([^>]*)>\^/\^\1\2\3 <VS-hSS\5\2-lI-fNIL \4> <SSmod-lI-f\6\7 <: :> <SS-lI-f\6\7\5\8>>\^/ ||
        # branch off final SBAR as extraposed modifier Cr rel clause:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gCr ...> <Cr WH# ... t# ...>}
        (s/\^([SQCFVIBLAGR])([^ ]*?)($CODA) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 $SUITABLE_REL_CLAUSE)>\^/\^\1\2\3 <\1\2-h\{C-rN}\5-lI-fNIL \4> <C-rN-lN-f\6>\^/ && ($j=5)) ||
        # branch off final SBAR as extraposed modifier IP:                           {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-gRP ...> <IP WH# ... t# ...>}
        # try not to take [AR]-aN-x by having [^ x]* instead of normal [^ ]*
        (s/\^([SQCFVIBLAGR])([^ x]*?)($CODA) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*) \[ADVP \[-NONE- \*T\*\7\]\]([^>]*)\]>\^/\^\1\2\3 <\1\2-h\{I-aN}\5-lI-fNIL \4> <I-aN-lN-f\6 \8\9>\^/ && ($j=6)) ||
        # branch off final SBAR as extraposed modifier CS complement:                {VP ... t#' ... <SBAR#' [WH# which/who/that/nil] ... t# ...>} => {<VP-g{C-rN} ...> <CS ...>}
        (s/\^([SQCFVIBLAGRN])([^ ]*?)($CODA) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(SBAR[^ ]*\5 \[(?:IN|DT) that\][^>]*)>\^/\^\1\2\3 <\1\2-hC\5-lI-fNIL \4> <C-lN-f\6>\^/ && ($j=7)) ||
        # branch off final PP as extraposed modifier:
        (s/\^([SQCFVIBLAGRN])([^ ]*?)($CODA) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <([VP]P[^ ]*\5 (?!\[IN of)[^>]*)>\^/\^\1\2\3 <\1\2-h\{R-aN}\5-lI-fNIL \4> <R-aN-lN-f\6>\^/ && ($j=7)) ||
        # inverted sentence: branch off final raised complement SS (possibly quoted)
        (s/\^(S)([^ ]*?)($CODA) (<.*\[-NONE- \*ICH\*(-[0-9]+)\].*) <(S)([^ ]*)\5([^>]*)>\^/\^\1\2\3-modeverused? <V-hS\5\2-lI-fNIL \4> <S-lN-f\6\7\5\8>\^/ && ($j=8)) ||

        # branch off final parenthetical sentence wihtout extraction
        (s/\^(S|V\-iN|Q|[VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) (<.*) (<-L.B- -L.B->) <(S-TOBE([VIBA])S[^ ]*) ([^>]*)> (<-R.B- -R.B->)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\7-lN-f\6 \5 \8 \9>\^/ && ($j=113)) ||

        # branch off final punctuation (passing -o to head). No fire on this gcg13. Lots of hit on Gcg1 version. Why? Because of the 144 and 154.5 rules?
#        (s/\^(N(?!-a))([^ ]*?)($CODA?)N-aD (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>\^/\^\1\2\3N <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>\^/ && ($j=8.5)) ||
#        (s/\^(SS|VS\-iNS|QS|CS|ES|Cr|RC|[VIBLAGR]|A-aN-x|R-aN-x|NS|NP)([^ ]*?)($CODA) (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>\^/ && ($j=9)) ||
#        (s/\^([SQCEVIBLAGR](?!-a)r?|[VIBLAGR]-aNr?|[AR]-aN-x|N)([^ ]*?)($CODA) (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>\^/ && ($j=9)) ||
        (s/\^([SQCFVIBLAGRN])([^ ]*?)($CODA) (<.*) <(?!POS)([^ ]*) ($FINAL_PUNCT)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\5-lM-f\5 \6>\^/ && ($j=9)) ||

        # branch off final possessive 's
        (s/\^(D|N(?!-a))([^ ]*) (<.*) <(POS 's?)>\^/\^\1\2 <N-lA-fNIL \3> <D-aN-lI-f\4>\^/ && ($j=10)) ||

        # branch off final 'that S'
        (s/\^([^ ]*)($CODA) (<.*) (<IN that> <S[^>]*>)\^/<\1\2 \3 <SBAR \4>>/ && ($j=10.5)) ||

        ######################################################################
        ## 3. HIGH BRANCHING / LOW PRECEDENCE INITIAL CONSTITUENTS

        #### within conjunction
        # branch off initial semicolon delimiter
        (s/\^([^ ]*)-pPs-c(\{?\1}?)([^ ]*?)($CONJCODA) <([^ ]*) (;)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;).*)\^/\^\1-pPs-c\2\3\4 <\5-lM-f\5 \6> <\1-c\2-pPs\3-lI-fNIL \7>\^/ && ($j=70)) ||
        # branch off initial comma delimiter
        (s/\^([^ ]*)-pPc-c(\{.*}|[^- ]*)([^ ]*?)($CONJCODA) <([^ ]*) (,)> (.*<(?:CC|ADVP \[RB then|ADVP \[RB not).*)\^/\^\1-pPc-c\2\3\4 <\5-lM-f\5 \6> <\1-c\2-pPc\3-lI-fNIL \7>\^/ && ($j=71)) ||
        # branch off initial conj delimiter and final conjunct (and don't pass -p down)
        (s/\^([^ ]*)(-c[^ ]*-pP[cs])($VARCONJCODA)(-[fl][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*)> (<.*)\^/\^\1\2\3\4 <X-cX-dX-lI-f\5> <\1\3-lC-fNIL \6>\^/ && ($j=72)) ||
        # branch off initial conj delimiter and final conjunct (no -p to remove)
        (s/\^([^X][^ ]*)((-pP[cs])?-c\{?\1}?)([^ ]*?)($CONJCODA) <((?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*)> (<.*)\^/\^\1\2\4\5 <X-cX-dX-lI-f\6> <\1\4-lC-fNIL \7>\^/ && ($j=73)) ||
#        (s/\^([^X][^ ]*?)(-c{?\1}?)([^ ]*?)(-[fghjlp][^ ]*) <((?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*)> (<.*)\^/\^\1\2\3\4 <X-cX-dX-lI-f\5> <\1\3-lC-fNIL \6>\^/ && ($j=73)) ||
#        # branch off initial comma/semi/colon/dash between matching constituents
#        s/\^(C)([^sc][^- ]*)([^ ]*?)()(-[fghjlp][^ ]*) (<[^ ]* (?:,|;|:|--|-)>) (<.*)\^/\^\1mod\2\3\4\5 \6 <\2\4-lI-fNIL \7>\^/ ||
        # branch off initial conjunct prior to semicolon delimiter
#        (s/\^([^ ]*?)(-c({?\1}?)-pPs)(-[fghjlp][^ ]*) (<.*?) (<[^ ]* ;> <.*)\^/\^\1\2\4 <\1-lC-fNIL \5> <\1-pPs-c\3-lI-fNIL \6>\^/ && ($j=74)) ||
        (s/\^([^ ]*?)(-c\{?\1}?)(-pPs)($CONJCODA) (<.*?) (<[^ ]* ;> <.*)\^/\^\1\2\3\4 <\1-lC-fNIL \5> <\1\3\2-lI-fNIL \6>\^/ && ($j=74)) ||
        # branch off initial conjunct prior to comma delimiter
#        (s/\^([^ ]*?)(-c({?\1}?)-pPc)(-[fghjlp][^ ]*) (<.*?) (<[^ ]* ,> <.*)\^/\^\1\2\4 <\1-lC-fNIL \5> <\1-pPc-c\3-lI-fNIL \6>\^/ && ($j=75)) ||
        (s/\^([^ ]*?)(-c\{?\1}?)(-pPc)($CONJCODA) (<.*?) (<[^ ]* ,> <.*)\^/\^\1\2\3\4 <\1-lC-fNIL \5> <\1\3\2-lI-fNIL \6>\^/ && ($j=75)) ||
        # branch off initial conjunct prior to conj delimiter (and don't pass -p down)
#        (s/\^([^ ]*?)-c(({?\1}?)(-pP[sc]))(-[ghj][^ ]*?)(-[fl][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)\^/\^\1-c\2\5\6 <\1\5-lC-fNIL \7> <\1\4-c\3\5-lI-fNIL \8>\^/ && ($j=76)) ||
        (s/\^([^ ]*?)(-c\{?\1}?)(-pP[sc])($VARCONJCODA)(-[fl][^ ]*) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)\^/\^\1\2\3\4\5 <\1\4-lC-fNIL \6> <\1\3\2\4-lI-fNIL \7>\^/ && ($j=76)) ||
        # branch off initial conjunct prior to conj delimiter (no -p to remove). Split rule 77 into 3 rules
        # CcVP-pC -> VP-cVP-pPc and CsVP-pS -> VP-cVP-pPs
        (s/\^([^ ]*?)(-c\{?\1}?)(-pP[sc])([^ ]*?)($CODA) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)\^/\^\1\2\3\4\5 <\1-lC-fNIL \6> <\1\3\2-lI-fNIL \7>\^/ && ($j=77.7)) ||
        # CcVP -> VP-pPc-cVP  and CsVP -> VP-pPs-cVP
        (s/\^([^ ]*?)(-pP[sc]-c\1)([^ ]*?)($CODA) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)\^/\^\1\2\3\4 <\1\3-lC-fNIL \5> <\1\2\3-lI-fNIL \6>\^/ && ($j=77.8)) ||
        # CVP -> VP-cVP
        (s/\^([^ ]*?)(-c\1)([^ ]*?)($CODA) (<.*?) (<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)\^/\^\1\2\3\4 <\1\3-lC-fNIL \5> <\1\2\3-lI-fNIL \6>\^/ && ($j=77)) ||

        # branch off initial punctuation (passing -o to head)
#        (s/\^(SS|VS\-iNS|QS|CS|ES|Cr|RC|[VIBLAGR][SP]r?|NS|A-aN-x|R-aN-x|NP)([^ ]*?)($CODA) <([^ ]*) ($INIT_PUNCT)> (<.*)\^/\^\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>\^/ && ($j=11)) ||
        (s/\^([SQCFVIBLAGRN](?!-[ar])|[VIBLAGRN]-a.(?!-x)|A-aN-x|R-aN-x)((?!-[cp])[^ ]*?)($CODA) <([^ ]*) ($INIT_PUNCT)> (<.*)\^/\^\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>\^/ && ($j=11)) ||

        #### rel clause
        # implicit-pronoun relative: delete initial empty interrogative phrase
        (s/\^([CV](?:-rN))([^ ]*?)(-[fghjlirp][^ ]*) <WHNP[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)\^/\^\1\2\3 <V\2-gN\4-lI-fNIL \5>\^/ && ($j=56)) ||
        # implicit-pronoun relative: delete initial empty interrogative phrase as adverbial
        (s/\^([CV](?:-rN))([^ ]*?)(-[fghjlrp][^ ]*) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)\^/\^\1\2\3 <V\2-g\{R-aN}\4-lI-fNIL \5>\^/ && ($j=57)) ||
        # branch off initial relative noun phrase
        (s/\^([CV](?=-rN)|R-aN(?!-x))(?:-rN)?([^ ]*?)($CODA) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1-rN\2\3 <N-rN-lN-f\4\5\6\7> <V\2-gN\6-lI-fNIL \8>\^/ && ($j=58)) ||
        # branch off initial relative adverbial phrase with empty subject ('when in rome')
        (s/\^([CV](?=-rN)|R-aN(?!-x))(?:-rN)?([^ ]*?)($CODA) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (.*\[-NONE- *\*T\*\6\].*)>\^/\^\1-rN\2\3 <R-aN-rN-lN-f\4\5\6\7> <A-aN\2-g\{R-aN}\6-lI-fNIL \8>\^/ && ($j=59)) ||
        # branch off initial relative adverbial phrase
        (s/\^([CV](?=-rN)|R-aN(?!-x))(?:-rN)?([^ ]*?)($CODA) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1-rN\2\3 <R-aN-rN-lN-f\4\5\6\7> <V\2-g\{R-aN}\6-lI-fNIL \8>\^/ && ($j=60)) ||
        # embedded question: branch off initial interrogative RP whether/if
        (s/\^([CV]-rN)([^ ]*?)(?:-rN)?($CODA) <(IN[^>]*)> (<.*)\^/\^\1\2\3 <R-aN-iN-lN-f\4> <V\2-g\{R-aN}-lI-fNIL \5>\^/ && ($j=61)) ||

        #### AP|RP
        # branch off initial specifier NS measure
        (s/\^([AR]-aN(?!-x))((?!-[cp])[^ ]*?)($CODA) <NP([^>]*)> (<.*)\^/\^\1\2\3 <N-lA-fNS\4> <\1\2-aN-lI-fNIL \5>\^/ && ($j=14)) ||
        # delete initial empty subject. No fire on this gcg13. No fire on Gcg1 version, either. Why? 
        (s/\^(R-aN(?!-x))((?!-[cp])[^ ]*?)($CODA) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)\^/<\1\2\3 \4>/ && ($j=15)) ||
#        # branch off initial modifier R-aN-x
#        s/\^(RP)([^ ]*?)($CODA) <(RB|ADVP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO|[^ ]*-PRD).*)\^/\^\1\2\3 <R-aN-x-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ ||
#        s/\^(AP|RP)([^ ]*?)($CODA) <(RB|ADVP|ADJP)([^>]*)> ([^\|]*<(?:JJ|ADJP|VB|VP|RB|IN|TO).*)\^/\^\1\2\3 <R-aN-x-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ ||
##           s/\^(AP|RP)([^ ]*)(-f[^ ]*) (<TO.*<VP.*)\^/\^\1\2\3 <IP\2-fNIL \4>\^/ ||
        # for good / for now
        (s/\^(R-aN(?!-x))((?!-[cp])[^ ]*?)($CODA) <(IN[^>]* for)> <(?![^>]* long[^>]*)(?![^>]* awhile[^>]*)(RB[^>]*|ADVP[^ ]* \[RB[^>]*)>\^/\^\1\2\3 <R-aN-bN-lI-f\4> <N-lA-f\5>\^/ && ($j=16)) ||

        #### initial filler
        # content question: branch off initial interrogative NS
        (s/\^(S)((?!-[cp])[^ ]*?)($CODA) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <N-iN-lN-f\4\5\6\7> <Q\2-gN\6-lI-fNIL \8>\^/ && ($j=17)) ||
        # content question: branch off initial interrogative RP
        (s/\^(S)((?!-[cp])[^ ]*?)($CODA) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <R-aN-iN-lN-f\4\5\6\7> <Q\2-g\{R-aN}\6-lI-fNIL \8>\^/ && ($j=18)) ||
        # topicalized sentence: branch off initial topic SS (possibly quoted)
        (s/\^([SCFV](?!-a))((?!-[cp])[^ ]*?)($CODA) (?!<[^ ]*-SBJ)<(S)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <S-lN-f\4\5\6\7> <V\2-gS\6-lI-fNIL \8>\^/ && ($j=19)) ||
        # topicalized sentence: branch off initial topic NS   ***<[^ ]* \[-NONE- [^\]]*\]>|
        (s/\^([SCFV](?!-a))((?!-[cp])[^ ]*?)($CODA) (?!<[^ ]*-SBJ)<(NP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <N-lN-f\4\5\6\7> <V\2-gN\6-lI-fNIL \8>\^/ && ($j=20)) ||
        # expletive there
        (s/\^([SCFV](?!-a))((?!-[cp])[^ ]*?)($CODA) <NP[^ ]* \[EX ([^\]\[]*)\]> (<.*)\^/\^\1\2\3 <A-aN-lA-fEX \4> <V\2-a\{A-aN}-lI-fNIL \5>\^/ && ($j=21)) ||
# SAI    # topicalized sentence: branch off initial topic AP
# SAI    (s/\^([SCEV](?!-a))((?!-[cp])[^ ]*?)($CODA) (?!<[^ ]*-SBJ)<(ADJP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<VP.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <A-aN-lN-f\4\5\6\7> <Q\2-g{A-aN}\6-lI-fNIL \8>\^/ && ($j=21)) ||
        # topicalized sentence: branch off initial topic AP
        (s/\^([SCFV](?!-a))((?!-[cp])[^ ]*?)($CODA) (?!<[^ ]*-SBJ)<(ADJP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <A-aN-lN-f\4\5\6\7> <V\2-g\{A-aN}\6-lI-fNIL \8>\^/ && ($j=21.5)) ||
        # topicalized sentence: branch off initial topic RP
        (s/\^([SCFV](?!-a))((?!-[cp])[^ ]*?)($CODA) (?!<[^ ]*-SBJ)<((?!WH)[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (<.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <R-aN-lN-f\4\5\6\7> <V\2-g\{R-aN}\6-lI-fNIL \8>\^/ && ($j=22)) ||
        # embedded sentence: delete initial empty complementizer
        (s/\^(C)((?!-[cp])[^ ]*?)($CODA) <-NONE-[^>]*> (<.*)\^/\^\1\2\3 <V\2-lI-fNIL \4>\^/ && ($j=23)) ||
        # embedded sentence: branch off initial complementizer
#        s/\^(V|I)E([^ ]*?)($CODA) <(IN[^>]*)> (<.*)\^/\^\1E\2\3 <\1E\2-b\1S-lM-f\4> <\1S-lI-fNIL \5>\^/ ||
        (s/\^C((?!-[cp])[^ ]*?)($CODA) <(IN[^>]*)> (<.*)\^/\^C\1\2 <C\1-bV-lI-f\3> <V-lA-fNIL \4>\^/ && ($j=24)) ||
        (s/\^F((?!-[cp])[^ ]*?)($CODA) <(IN[^>]*)> (<.*)\^/\^F\1\2 <F\1-bI-lI-f\3> <I-lA-fNIL \4>\^/ && ($j=25)) ||
        # embedded noun: branch off initial preposition
#        s/\^(N)E([^ ]*?)(-[ir][A-Z])?($CODA) <(IN[^>]*)> (<.*)\^/\^\1E\2\3\4 <\1E\2-b\1P-lM-f\5> <\1P\3-lI-fNIL \6>\^/ ||
        (s/\^O((?!-[cp])[^ ]*?)(-[ir][A-Z]+)?($CODA) <(IN[^>]*)> (<.*)\^/\^O\1\2\3 <O\1-bN-lI-f\4> <N\2-lA-fNIL \5>\^/ && ($j=26)) ||
        # embedded question: branch off initial interrogative RP whether/if
        (s/\^(V\-iN)([^ ]*?)($CODA) <(IN[^>]*)> (<.*)\^/\^\1\2\3 <R-aN-iN-lN-f\4> <V\2-g\{R-aN}-lI-fNIL \5>\^/ && ($j=27)) ||

        #### initial RP/RC modifier
        # branch off initial modifier RP with colon
        (s/\^([SQCFVIBLAG](?!-a))((?!-[cp])[^ ]*?)($CODA) <(PP|RB|ADVP|CC|FRAG|NP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> <([^ ]*) (:)> (<.*)\^/\^\1\2\3 <R-aN-lM-fNIL <R-aN-lI-f\4\5> <\6 \7>> <\1\2-lI-fNIL \8>\^/ && ($j=28)) ||
        # branch off initial modifier RP IP
        (s/\^([SQCFVIBLAG](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) <S(?![^ ]*-SBJ)[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ($NO_AP_HEAD\[TO[^>]*)> (?!.*<CC.*\^)(<.*)\^/\^\1\2\3 <R-aN-lM-fNIL <I-aN-lI-fNIL \4>> <\1\2-lI-fNIL \5>\^/ && ($j=29)) ||
        # branch off initial modifier RP AP
        (s/\^([SQCFVIBLAG](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) <S(?![^ ]*-SBJ)[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]\] ([^>]*)> (?!.*<CC.*\^)(<.*)\^/\^\1\2\3 <R-aN-lM-fNIL \4> <\1\2-lI-fNIL \5>\^/ && ($j=30)) ||
        # branch off initial modifier RP from SBAR
        (s/\^([SQCFVIBLAG](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) <(SBAR(?![^ ]*-SBJ)[^ ]* (?!\[IN that|\[IN for|\[IN where|\[IN when)(?!\[WH[^  ]*))([^>]*)> (?!.*<CC.*<SBAR.*\^)(<.*)\^/\^\1\2\3 <R-aN-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=31)) ||
        # branch off initial modifier RC from SBAR-ADV
        (s/\^([SQCFVIBLAG](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) <(SBAR(?:-ADV|-TMP)[^ ]* \[WH)([^>]*)> (<.*)\^/\^\1\2\3 <V-rN-lN-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=32)) ||
        # branch off initial RB + JJS as modifier RP  (e.g. "at least/most/strongest/weakest")
        (s/\^([SQCFVIBLAGN](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) (<IN[^>]*> <JJ[^>]*>) (?!<CC)(<.*)\^/\^\1\2\3 <R-aN-lM-fNIL \4> <\1\2-lI-fNIL \5>\^/ && ($j=33)) ||
        # branch off initial modifier RP  (incl determiner, e.g. "both in A and B")
#WORSE:        s/\^(SS|VS\-iNS|QS|CS|ES|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)($CODA) <([^ ]*-TMP)([^>]*)> (?!<CC)(<.*)\^/\^\1\2\3 <RP-t-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ ||
        (s/\^([SQCFVIBLAG](?!-a)|[VIBLG]-aN)((?!-[cp])[^ ]*?)($CODA) <(NP-ADV|NP-TMP)([^>]*)> (?!<CC)(<.*)\^/\^\1\2\3 <R-aN-lM-fNS-TMP\5> <\1\2-lI-fNIL \6>\^/ && ($j=33.5)) ||
        (s/\^(C(?!(?:-)?r)|[SQFVIBLAG](?!-a)|[VIBLG]-aN|[VIBLG]-a\{A-aN})((?!-[cp])[^ ]*?)($CODA) <(DT|PP|RB|IN|ADVP|CC|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)> (?!<CC)(<.*)\^/\^\1\2\3 <R-aN-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=34)) ||
        (s/\^(N(?!-aD|-[cp]))([^ ]*?)(-[ir][A-Z]+)?([^ \}]*?)($CODA) <(CC)([^>]*)> (<(?!PP|WHPP).*)\^/\^\1\2\3\4\5 <R-aN-x-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>\^/ && ($j=35)) ||
        (s/\^(N(?!-aD|-[cp]))([^ ]*?)(-[ir][A-Z]+)?([^ \}]*?)($CODA) <(RB|PDT|(?:CC|DT) (?:[Nn]?[Ee]ither|[Bb]oth)(?=.*<CC.*\^)|DT (?:[Aa]ll|[Bb]oth|[Hh]alf)|(?:ADJP|QP)(?=[^>]*\[DT [^\]]*\]>))([^>]*)> (<.*)\^/\^\1\2\3\4\5 <R-aN-x-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>\^/ && ($j=36)) ||
        (s/\^(N(?!-aD|-[cp]))([^ ]*?)(-[ir][A-Z]+)?([^ \}]*?)($CODA) <(ADVP|PP)([^>]*)> (?!<CC)(<.*)\^/\^\1\2\3\4\5 <R-aN-lM-f\6\7> <\1\2\3\4-lI-fNIL \8>\^/ && ($j=37)) ||
#        s/\^(SS|VS\-iNS|QS|CS|ES|[VIBLAG]S|VP|IP|BP|LP)([^ ]*?)($CODA) <(S)([^ ]* \[NP[^ ]* \[-NONE- \*-[^>]*)> (<.*)\^/\^\1\2\3 <RP-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ ||
        # branch off initial modifier R-aN-x-iNS/R of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        (s/\^([AR]-aN(?!-x|-[cp]))([^ ]*?)(-[ir][A-Z]+)($CODA) <(WRB)([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)\^/\^\1\2\3\4 <R-aN-x\3-lM-f\5\6> <\1\2-lI-fNIL \7>\^/ && ($j=38)) ||
        # branch off initial modifier R-aN-x of AP/RP  (incl determiner, e.g. "both in A and B") (only if better head later in phrase) (passing -o to head)
        (s/\^([AR]-aN(?!-x|-[cp]))([^ ]*?)($CODA) <(DT|PP|RB|IN(?=[^\|]*<IN)|ADVP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (?!<CC)([^\|]*<(?:JJ|ADJP|VB|VP|RB|WRB|IN|TO|[^ ]*-PRD).*)\^/\^\1\2\3 <R-aN-x-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=39)) ||

        #### CODE REVIEW: WHADVP/WP$ in NS needs to inherit -o  ******************

        #### sentence types
        # branch off initial parenthetical sentence with extraction
        (s/\^([SQCFVIBLAGR](?:-aN)?)((?!-[cp])[^ a]*?)($CODA) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)\^/\^\1\2\3 <V-gS\6-lN-f\4\5> <\1\2-lI-fNIL \7>\^/ && ($j=40)) ||
        # branch off initial parenthetical sentence w/o extraction
        (s/\^(V-aN(?!-x))((?!-[cp])[^ ]*?)($CODA) <(S(?![A-Z]))([^>]*)> (.*<VP.*)\^/\^\1\2\3 <V-lN-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=41)) ||
        # imperative sentence: delete empty NS
        (s/\^(S)((?!-[cp])[^ ]*?)($CODA) <NP[^ ]* \[-NONE- \*\]> (<VP$NO_VP_HEAD \[VB.*)\^/\^\1\2\3 <B-aN\2-lI-fNIL \4>\^/ && ($j=42)) ||
        # declarative (inverted or uninverted) sentence: unary expand to VS. Can't match |<VP.*<(?:NP|[^ ]*-SBJ) because there's no VP-SBJ and no '>' before the ending ^. Tried adding .* before ending ^ doesn't help
#        (s/\^(S|C)((?!-[cp])[^ ]*?)($CODA) (<(?:NP|[^ ]*-SBJ).*<VP.*|<VP.*<(?:NP|[^ ]*-SBJ))\^/\^\1\2\3 <V\2-lI-fNIL \4>\^/ && ($j=43)) ||
        (s/\^(S|C)((?!-[cp])[^ ]*?)($CODA) (<(?:NP|[^ ]*-SBJ).*<VP.*)\^/\^\1\2\3 <V\2-lI-fNIL \4>\^/ && ($j=43)) ||
        # it-extraposition normalization to VP
        (s/\^(V)(-aNe-b(?:C|F|\{I-aN}|\{V-iN}|N))($CODA) (.*)\^/\^\1\2\3 <\1-aN-lI-fNIL \4>\^/ && ($j=43.5)) ||
        # polar question: unary expand to QS
        (s/\^(S|V\-iN)((?!-[cp])[^ ]*?)($CODA) (<[^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Hh]as|[Hh]ave|[Hh]ad)).*<NP.*)\^/\^\1\2\3 <Q\2-lI-fNIL \4>\^/ && ($j=44)) ||
        # imperative sentence: unary expand to BP   ***PROBABLY NULL CAT HERE***
        (s/\^(S)((?!-[cp])[^ ]*?)($CODA) (<VP$NO_VP_HEAD \[VB.*)\^/\^\1\2\3 <B-aN\2-lI-fNIL \4>\^/ && ($j=45)) ||
        # embedded question: branch off initial interrogative NS and final modifier IP with NS gap (what_i to find a picture of t_i)
        (s/\^(V\-iN)([^ ]*?)($CODA) <(WHNP[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>\^/\^\1\2\3 <N-iN-lN-f\4\5\6> <I-aN-gN\5-lI-fNIL \7>\^/ && ($j=46)) ||
        # embedded question: branch off initial interrogative RP and final modifier IP with RP gap (how_i to find a picture t_i)
        (s/\^(V\-iN)([^ ]*?)($CODA) <(WH[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>\^/\^\1\2\3 <R-aN-iN-lN-f\4\5\6> <I-aN-g\{R-aN}\5-lI-fNIL \7>\^/ && ($j=47)) ||
        # embedded question: branch off initial interrogative NS
        (s/\^(V\-iN)([^ ]*?)($CODA) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <N-iN-lN-f\4\5\6\7> <V\2-gN\6-lI-fNIL \8>\^/ && ($j=48)) ||
        # embedded question: branch off initial interrogative RP
        (s/\^(V\-iN)([^ ]*?)($CODA) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <R-aN-iN-lN-f\4\5\6\7> <V\2-g\{R-aN}\6-lI-fNIL \8>\^/ && ($j=49)) ||
        # nom clause: branch off initial interrogative NS and final modifier IP with NS gap (what_i to find a picture of t_i)
        (s/\^(N(?!-aD))([^ ]*?)($CODA) <(WHNP[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>\^/\^\1\2\3 <N\2-b\{I-aN-gN}-lI-f\4\5\6> <I-aN-gN\5-lA-fNIL \7>\^/ && ($j=46)) ||
        # nom clause: branch off initial interrogative RP and final modifier IP with RP gap (how_i to find a picture t_i)
        (s/\^(N(?!-aD))([^ ]*?)($CODA) <(WH[^ ]*)(-[0-9]+)([^>]*)> <S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\5\][^>]*)>\^/\^\1\2\3 <N\2-b\{I-aN-g\{R-aN}}-lI-f\4\5\6> <I-aN-g\{R-aN}\5-lA-fNIL \7>\^/ && ($j=47)) ||
        # nom clause: branch off initial interrogative NS
        (s/\^(N(?!-aD))([^ ]*?)($CODA) <(WHNP)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <N\2-b\{V-gN}-lI-f\4\5\6\7> <V-gN\6-lA-fNIL \8>\^/ && ($j=48)) ||
        # nom clause / nom clause modifier: branch off initial interrogative RP
        (s/\^(N(?!-aD))([^ ]*?)($CODA) <(WH[A-Z]*)([^ ]*)(-[0-9]+)((?![0-9])[^>]*)> (.*\[-NONE- *\*T\*\6\].*)\^/\^\1\2\3 <N\2-b\{V-g\{R-aN}}-lI-f\4\5\6\7> <V-g\{R-aN}\6-lA-fNIL \8>\^/ && ($j=49)) ||
#        # polar question: branch off initial BP-taking auxiliary
#        (s/\^(Q)([^ ]*?)($CODA) <([^\]]*(?:MD|VB[A-Z]? (?:[Dd]oes|[Dd]o|[Dd]id|'d))[^>]*)> (<.*)\^/\^\1\2\3 <\1\2-bB-lM-f\4> <B-lI-fNIL \5>\^/ && ($j=50)) ||
#        # polar question: branch off initial NS-taking auxiliary
#        (s/\^(Q)([^ ]*?)($CODA) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<(?:NP|NN|DT)[^>]*>)\^/\^\1\2\3 <\1\2-bN-lM-f\4> <N-lI-fNIL \5>\^/ && ($j=51)) ||
#        # polar question: branch off initial AP-taking auxiliary
#        (s/\^(Q)([^ ]*?)($CODA) <([^\]]*VB[A-Z]? (?:[Ii]s|[Aa]re|[Ww]as|[Ww]ere|'s|'re)[^>]*)> (<.*)\^/\^\1\2\3 <\1\2-bA-lM-f\4> <A-lI-fNIL \5>\^/ && ($j=52)) ||
#        # polar question: branch off initial LP-taking auxiliary  ***NOTE: 's AND 'd WON'T GET USED***
#        (s/\^(Q)([^ ]*?)($CODA) <([^\]]*VB[A-Z]? (?:[Hh]as|[Hh]ave|[Hh]ad|'s|'ve|'d)[^>]*)> (<.*)\^/\^\1\2\3 <\1\2-bL-lM-f\4> <L-lI-fNIL \5>\^/ && ($j=53)) ||
        # polar question: allow subject gap without inversion
        (s/\^(Q)([^ ]*-gN(-[0-9]+)[^ ]*?)($CODA) <NP[^ ]* \[-NONE- \*T\*\3\]> (<.*)\^/\^\1\2\4 <V-aN-fNIL \5>\^/ && ($j=54)) ||
###        # polar question error: move extracted subject to inverted location -- NO: SHOULD NOT NEED AUX WHEN SUBJ EXTRACTED!
###        (s/\^(Q)([^ ]*-gN(-[0-9]+)[^ ]*?)($CODA) (<NP[^ ]* \[-NONE- \*T\*\3\]>) <VP[^ ]* (\[VB[^\]]*\]) (\[.*)>\^/<\1\2\4 \6 \5 \7>/ && ($j=54)) ||
        # embedded sentence: delete initial empty interrogative phrase     ****WHY WOULD THIS HAPPEN??? NO WH IN EMBEDDED SENTENCE****
        (s/\^(C)([^ ]*?)($CODA) <WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^>]*> (<.*)\^/\^\1\2\3 <V\2-gN\4-lI-fNIL \5>\^/ && ($j=55)) ||

        #### middle NS
        # branch off middle modifier AP colon
        (s/\^(N)([^ ]*?)($CODA) (<.*) (<[^ ]* :> <.*)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL \5>\^/ && ($j=62)) ||

        #### conjunction
        # branch final right-node-raising complement NS
        (s/\^([^ ]*?)($CONJCODA) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <NP([^ ]*\4[^>]*)>\^/\^\1\2 <\1-hN\4-lI-fNIL \3> <N-lN-fNS\5>\^/ && ($j=63)) ||
        # branch final right-node-raising modifier AP
        (s/\^([^ ]*?)($CONJCODA) (<.*\[-NONE- \*RNR\*(-[0-9]*)\].*) <((?:PP)[^ ]*\4[^>]*)>\^/\^\1\2 <\1-lI-fNIL \3> <A-aN-lM-f\5>\^/ && ($j=64)) ||
        # pinch ... CC ... -NONE- and re-run
        (s/\^([^C][^ ]*?)($CONJCODA)(?!.*\|) (<.*) (<CC[^>]*>) (<.*) (<[^ ]* \[-NONE- [^\]]*\]>)\^/<\1\2 <\1 \3 \4 \5> \6>/ && ($j=65)) ||
        # branch off initial colon in colon...semicolon...semicolon construction
        (s/\^([^ ]*)($CONJCODA)(?!.*\|) (<. :>) <NP([^ ]*)([^>]*)> (<. ;.*<. ;.*)\^/\^\1\2 \3 <N\4-lA-fNIL <N\4\5> \6>\^/ && ($j=65.5)) ||
        (s/\^([^ ]*)($CONJCODA)(?!.*\|) (<. :>) <([^ ]*)([^>]*)> (<. ;.*<. ;.*)\^/\^\1\2 \3 <\4-lA-fNIL <\4\5> \6>\^/ && ($j=66)) ||
        # branch off initial conjunct prior to semicolon delimiter
        (s/\^([A-Z]-[abrik].(?:-x)?)([^ cp]*?)($CODA)(?!.*\|) (<.*?) (<[^ ]* ;> .*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)\^/\^\1\2\3 <\1\2-lC-fNIL \4> <\1\2-pPs-c\{\1\2}-lI-fNIL \5>\^/ && ($j=66.6)) ||
        (s/\^([^ \-cp]*)($CODA)(?!.*\|) (<.*?) (<[^ ]* ;> .*<(?:CC|ADVP \[RB then|ADVP \[RB not|. ;)[^>]*> <.*)\^/\^\1\2 <\1-lC-fNIL \3> <\1-pPs-c\1-lI-fNIL \4>\^/ && ($j=67)) ||
        # branch off initial conjunct prior to comma delimiter
        (s/\^([A-Z]-[abrik].(?:-x)?)([^ cp]*?)($RCONJCODA)(?!.*\|) (<.*?) (<[^ ]* ,> .*<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)\^/\^\1\2\3 <\1\2-lC-fNIL \4> <\1\2-pPc-c\{\1\2}-lI-fNIL \5>\^/ && ($j=67.6)) ||
        (s/\^([^ \-cp]*)($RCONJCODA)(?!.*\|) (<.*?) (<[^ ]* ,> .*<(?:CC|ADVP \[RB then|ADVP \[RB not)[^>]*> <.*)\^/\^\1\2 <\1-lC-fNIL \3> <\1-pPc-c\1-lI-fNIL \4>\^/ && ($j=68)) ||
        # branch off initial conjunct prior to conj delimiter
        (s/\^([A-Z]-[abrik].(?:-x)?)([^ cp]*?)($RCONJCODA)(?!.*\|) (<.*?) (<CC[^>]*> <.*)\^/\^\1\2\3 <\1\2-lC-fNIL \4> <\1\2-c\{\1\2}-lI-fNIL \5>\^/ && ($j=68.8)) ||
        (s/\^([^ \-cp]*)(-[fghjlr](?![^ ]*}})[^ cp]*)(?!.*\|) (<.*?) (<CC[^>]*> <.*)\^/\^\1\2 <\1-lC-fNIL \3> <\1-c\1-lI-fNIL \4>\^/ && ($j=69)) ||
#        # branch off initial conjunct prior to comma/semi/colon/dash between matching constituents
#        s/\^([^C][A-Z]S[^ ]*?)(-[fghjlp][^ ]*) <([^- ]*)([^>]*)> (<[^ ]* ,> <\3[^>]*>)\^/\^\1mod\2 <\1-lC-f\3\4> <C\1-lI-fNIL \5>\^/ ||

        ######################################################################
        ## 4. LOW BRANCHING / HIGH PRECEDENCE FINAL CONSTITUENTS

        # branch off final cleft SBAR as argument Cr:                                        {V-aNc ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {V-aNc <Nc ...> <Cr WH# ... t# ...>}
        (s/\^([SQCFVIBLAGRN])-aNc([^ ]*?)($CODA) (<.*) <(SBAR$SUITABLE_REL_CLAUSE)>\^/\^\1-aNc\2\3 <\1-aN\2-lI-fNIL \4> <C-rN-lA-f\5>\^/ && ($j=116)) ||

        # branch off final parenthetical sentence with extraction
        (s/\^([SQCFVIBLAGRN])([^ ]*?)($CODA) (<.*) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <V-gS\7-lN-f\5\6>\^/ && ($j=78)) ||

        # branch NS -> DS A-aN-x: 'the best' construction
        (s/\^(N(?!-a))([^ ]*?)($CODA) <(?:DT)([^>]*)> <(?:RB|ADJP)([^>]*)>\^/\^\1\2\3 <D-lA-f\4> <N-aD-lI-f\5>\^/ && ($j=79)) ||

        #### final VP|IP|BP|LP|AP (following auxiliary) -- raising verbs, pass subject to complement
        # branch off final VP as argument BP
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([ce]?)($CODA) (<.*(?:TO |MD | [Dd]o[\]>]| [Dd]oes[\]>]| [Dd]id[\]>]).*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<VP.*>)\^/\^\1\2\3 <\1-b\{B-aN}-lI-fNIL \4> <B-aN\2-lA-fNIL \5\6>\^/ && ($j=80)) ||
        # there propagation through BP
        (s/\^(Q|[VIBLGR])(-a\{A-aN\})([^ ]*?)($CODA) (<.*(?:TO |MD | [Dd]o[\]>]| [Dd]oes[\]>]| [Dd]id[\]>]).*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<VP.*>)\^/\^\1\2\3\4 <\1-aX-b\{B-aX\}-lI-fNIL \5> <B-a\{A-aN\}\3-lA-fNIL \6\7>\^/ && ($j=80.5)) ||
        # branch off final VP as argument LP (w. special cases b/c 's ambiguous between 'has' and 'is')
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([ce]?)($CODA) (.*(?: [Hh]ave| [Hh]aving| [Hh]as| [Hh]ad| 've|VBD 'd)>.*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<VP.*>)\^/\^\1\2\3 <\1-b\{L-aN}-lI-fNIL \4> <L-aN\2-lA-fNIL \5\6>\^/ && ($j=81)) ||
        (s/\^(Q|[VIBLAG]-aN(?!-x))([ce]?)($CODA) (.*<VBZ *'s>.*?) (<RB.*)?(<VP[^\]]* (?:$LP_FIRST).*>)\^/\^\1\2\3 <\1-b\{L-aN}-lI-fNIL \4> <L-aN\2-lA-fNIL \5\6>\^/ && ($j=82)) ||
        # there propagation through LP
        (s/\^(Q|[VIBLGR])(-a\{A-aN\})([^ ]*?)($CODA) (.*(?: [Hh]ave| [Hh]aving| [Hh]as| [Hh]ad| 've|VBD 'd)>.*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<VP.*>)\^/\^\1\2\3\4 <\1-aX-b\{L-aX\}-lI-fNIL \5> <L-a\{A-aN\}\3-lA-fNIL \6\7>\^/ && ($j=82.3)) ||
        (s/\^(Q|[VIBLGR])(-a\{A-aN\})([^ ]*?)($CODA) (.*<VBZ *'s>.*?) (<RB.*)?(<VP[^\]]* (?:$LP_FIRST).*>)\^/\^\1\2\3\4 <\1-aX-b\{L-aX\}-lI-fNIL \5> <L-a\{A-aN\}\3-lA-fNIL \6\7>\^/ && ($j=82.6)) ||
        # branch off final PRT as argument P particle
        (s/\^(N(?!-a))([^ ]*?)($CODA) (<.*) <(PRT)([^ ]*) \[RP ([^ ]*)\]>\^/\^\1\2\3 <\1\2-kP\7-lI-fNIL \4> <P\7-lN-f\5\6 \7>\^/ && ($j=82.9)) ||
        (s/\^(Q|[VIBLAGRN](?=-a[ND](?!-x))|N(?!-a))([^ ]*?)($CODA) (<.*) <(PRT)([^ ]*) \[RP ([^ ]*)\]>\^/\^\1\2\3 <\1\2-bP\7-lI-fNIL \4> <P\7-lA-f\5\6 \7>\^/ && ($j=83)) ||
#        # branch off final modifier RP (extraposed from argument)    **TO PRESERVE EXTRAPOSN: /{\1\2\3 <\1\2-gRP\5-lI-fNIL \4> <RP-lM-f\5\6\7>\^/ ||
#        (s/\^(Q|[VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<VB.*<.*\[-NONE- \*ICH\*(-[0-9]+)(?![0-9]).*) <(VP[^ ]*)\5((?![0-9])[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-f\6\5\7>\^/ && ($j=84)) ||
        # branch off final NP-SBJ of subj-aux inverted clause as argument AP
# SAI    (s/\^(Q-b[^ ]*?)($CODA) (<.*) <(NP-SBJ)([^>]*)>\^/\^\1\2 <\1-bN-lI-fNIL \3> <N-lA-f\4\5>\^/ && ($j=84.5)) ||
        (s/\^(Q-b[^ ]*?)($CODA) (<.*) <(NP)([^>]*)>\^/\^\1\2 <\1-bN-lI-fNIL \3> <N-lA-f\4\5>\^/ && ($j=84.5)) ||
#        (s/\^(V-a\{.-aN\})($CODA) (<.*) <(NP)([^>]*)>\^/\^\1\2 <Q-b\{A-aN\}-bN-lI-fNIL \3> <N-lA-f\4\5>\^/ && ($j=84.5)) ||
        # there propagation
        (s/\^([VIBLG]-a\{.-aN\})($CODA) (<.*) <(NP(?!-(ADV|TMP|EXT|LOC|DIR|MNR|PRP|VOC))|[^ ]+-PRD)([^>]*)>\^/\^\1\2 <\1-bN-lI-fNIL \3> <N-lA-f\4\6>\^/ && ($j=84.5)) ||
        # branch off final VP|ADJP as argument AP
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([ce]?)($CODA) (.*(?: [Bb]e| [Bb]eing| [Bb]een| [Ii]s| [Ww]as|VBZ 's| [Aa]re| [Ww]ere| 're)>.*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<(?:VP|VB[DNG]|ADJP|JJ|CD|PP[^ ]*-PRD|IN|UCP|ADVP[^ ]*-PRD|SBAR[^ ]*-PRD (?!\[WH|\[IN that)).*>)\^/\^\1\2\3 <\1-b\{A-aN}-lI-fNIL \4> <A-aN\2-lA-fNIL \5\6>\^/ && ($j=85)) ||
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([ce]?)($CODA) (.*(?: [Bb]e| [Bb]eing| [Bb]een| [Ii]s| [Ww]as|VBZ 's| [Aa]re| [Ww]ere| 're)>.*?) ((?:<RB[^>]*> |<ADVP[^>]*> )*)(<(?![^ ]*-SBJ|[^ ]*-ADV)$NOMINALS[^>]*>)\^/\^\1\2\3 <\1-b\{A-aN}-lI-fNIL \4> <A-aN\2-lA-fNIL \5\6>\^/ && ($j=86)) ||
        (s/\^(N)([^ ]*?)($CODA) (.*(?: [Bb]e| [Bb]eing| [Bb]een| [Ii]s| [Ww]as|VBZ 's| [Aa]re| [Ww]ere| 're)>.*?) (<RB.*)?(<(?:VP|VB[DNG]|ADJP|JJ|CD|PP[^ ]*-PRD|IN|UCP|ADVP[^ ]*-PRD|SBAR[^ ]*-PRD (?!\[WH|\[IN that)).*>)\^/\^\1\2\3 <\1\2-b\{A-aN}-lI-fNIL \4> <A-aN-lA-fNIL \5\6>\^/ && ($j=85.5)) ||
        (s/\^(N)([^ ]*?)($CODA) (.*(?: [Bb]e| [Bb]eing| [Bb]een| [Ii]s| [Ww]as|VBZ 's| [Aa]re| [Ww]ere| 're)>.*?) (<RB.*)?(<(?:NP|NN|S[^ ]*-NOM|SBAR[^ ]*-NOM|SBAR[^ ]*-PRD).*>)\^/\^\1\2\3 <\1\2-b\{A-aN}-lI-fNIL \4> <A-aN-lA-fNIL <N-lI-fNIL \5\6>>\^/ && ($j=86.5)) ||
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([ce]?)($CODA) (<.*) <(ADJP|PRT|ADVP[^ ]*-PRD|PP[^ ]*-PRD|VP$NO_VP_HEAD \[VB[NG])([^>]*)>\^/\^\1\2\3 <\1-b\{A-aN}-lI-fNIL \4> <A-aN\2-lA-f\5\6>\^/ && ($j=87)) ||
        (s/\^([AR]-aN(?!-x))([^ ]*?)($CODA) (<IN[^>]*>) <(JJ)([^>]*)>\^/\^\1\2\3 <\1\2-b\{A-aN}-lI-fNIL \4> <A-aN-lA-f\5\6>\^/ && ($j=88)) ||
        # branch off final argument embedded question SS w. quotations
        (s/\^(Q|[VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<.*) <(SBARQ[^>]*)>\^/\^\1\2\3 <\1\2-bS-lI-fNIL \4> <S-lA-f\5>\^/ && ($j=89)) ||

        #### final NS
        # initiate passive - must have A/R-aN starting with VBN and first -NONE- in last <...> 
        ##(s/\^([AR]-aN[^-]*(?!-x))([^ ]*?)($CODA) (<VBN (?:[^<](?!-NONE-))* <[^>]* \[-NONE- \*(?:-[0-9]*)?[\*\]][^<]*)\^/\^\1\2\3 <L-aN\2-vN\3 \4>\^/ && ($j=90)) ||
        (s/\^([AR]-aN[^-]*(?!-x))([^ ]*?)($CODA) (<VBN [^>]*> <PP[^>]*> <S(?![^ ]*-PRP)[^ ]* \[NP[^ ]* \[-NONE- \*(?:-[0-9]*)?[\*\]].*)\^/\^\1\2\3 <L-aN\2-vN\3 \4>\^/ && ($j=89.5)) ||
        (s/\^([AR]-aN[^-]*(?!-x))([^ ]*?)($CODA) (<VBN [^>]*>( <[^\]]*>)* <[^>]* \[-NONE- \*(?:-[0-9]*)?[\*\]].*)\^/\^\1\2\3 <L-aN\2-vN\3 \4>\^/ && ($j=90)) ||
        (s/\^([AR]-aN-x)(?=[^ ]*-fVBN)([^ ]*?)($CODA) ([^>]*)\^/\^\1\2\3 <L-aN-bN-x\2-lI \4>\^/ && ($j=90.1)) ||
        #(s/\^([AR]-aN-x)(?=[^ ]*-fVBN)([^ ]*?)($CODA) ([^>]*)\^/\^\1\2\3 <L-aN-bN\2-lI \4>\^/ && ($j=90.1)) ||
        #(s/\^([AR]-aN[^-]*(?!-x))([^ ]*?)($CODA) (.*<VBN [^>]*>[^\]>]* \[[^ ]* \[-NONE- \*(?:-[0-9]*)?[\*\]]+.*)\^/\^\1\2\3 <L-aN\2-vN\3 \4>\^/ && ($j=90.5)) ||
        #(s/\^([AR]-aN[^-]*(?!-x))([^ ]*?)($CODA) (.*<VBN (?!.*\[VBN.*\[-NONE- \*(?:-[0-9]*)?[\*\]]*>\^).* \[-NONE- \*(?:-[0-9]*)?[\*\]]*>)\^/\^\1\2\3 <L-aN\2-vN\3 \4>\^/ && ($j=90)) ||
        ## delete final empty object of passive
        #(s/\^([AR]-aN(?!-x))([^ ]*?)($CODA) (<.*) <NP[^ ]* \[-NONE- \*(?:-[0-9]*)?\]>\^/<\1\2\3 \4>/ && ($j=90)) ||
        # delete final *PPA*- empty category
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<.*) <[^ ]* \[-NONE- \*PPA\*-[^ ]*\]>\^/<\1\2\3 \4>/ && ($j=91)) ||
        # branch off final IN|TO + NS as modifier RP
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<.*) (<(?:IN|TO)[^>]*> <NP[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=92)) ||
        # branch off final argument OS
        (s/\^(N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(WHPP[^ ]* \[IN of\][^>]*)>\^/\^\1\2\3\4 <\1\2-kO-lI-fNIL \5> <O\3-lN-f\6>\^/ && ($j=92.9)) ||
        (s/\^([VIBLAGRN]-a[ND](?!-x)|N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(WHPP[^ ]* \[IN of\][^>]*)>\^/\^\1\2\3\4 <\1\2-bO-lI-fNIL \5> <O\3-lA-f\6>\^/ && ($j=93)) ||
        # branch off final argument OS
        (s/\^(N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(PP[^ ]* \[IN of\][^>]*)>\^/\^\1\2\3\4 <\1\2\3-kO-lI-fNIL \5> <O-lN-f\6>\^/ && ($j=93.9)) ||
        (s/\^([VIBLAGRN]-a[ND](?!-x)|N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(PP[^ ]* \[IN of\][^>]*)>\^/\^\1\2\3\4 <\1\2\3-bO-lI-fNIL \5> <O-lA-f\6>\^/ && ($j=94)) ||
        # branch off final argument OS
        (s/\^(N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) (<IN of> <[^>]*>)\^/\^\1\2\3\4 <\1\2\3-kO-lI-fNIL \5> <O-lN-fNIL \6>\^/ && ($j=94.9)) ||
        (s/\^([VIBLAGRN]-a[ND](?!-x)|N(?!-a))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) (<IN of> <[^>]*>)\^/\^\1\2\3\4 <\1\2\3-bO-lI-fNIL \5> <O-lA-fNIL \6>\^/ && ($j=95)) ||
        # branch off final argument GP
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(S(?:[-=][^ ]*)?-NOM(?![^ ]*-TMP|[^ ]*-EXT)[^ ]*) \[NP-SBJ[^ ]* \[-NONE- [^\]]*\]\]([^>]*)>\^/\^\1\2\3\4 <\1\2-b\{G-aN}-lI-fNIL \5> <G-aN\3-lA-f\6\7>\^/ && ($j=96)) ||
        # branch off final argument GS
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(S(?:[-=][^ ]*)?-NOM)(?![^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2-bG-lI-fNIL \5> <G\3-lA-f\6\7>\^/ && ($j=97)) ||
        # branch off final argument NS
		# special handling for "no matter"
        (s/\^(R-aN(?!-x))([^ ]*?)(-i[^- \}]*)($CODA) (<DT [Nn]o>) <(NP|DT|NN|WHNP|S[^ ]*-NOM)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*matter)>\^/\^\1\2\3\4 <\1\2\3-bN-lI-fNIL \5> <N-lA-f\6\7>\^/ && ($j=97.4)) ||
        (s/\^([RA]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(NN)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2-bN-lI-fNIL \5> <N\3-lA-f\6\7>\^/ && ($j=97.45)) ||
        (s/\^([LVGB]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(NP)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2\3-bN-lI-fNIL \5> <N-lA-f\6\7>\^/ && ($j=97.46)) ||
        (s/\^([IAR]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(NP)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2-bN-lI-fNIL \5> <N\3-lA-f\6\7>\^/ && ($j=97.47)) ||
        (s/\^(R-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(DT|WHNP|S[^ ]*-NOM)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2-bN-lI-fNIL \5> <N\3-lA-f\6\7>\^/ && ($j=97.5)) ||
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(NP|DT|NN|WHNP|S[^ ]*-NOM)(?![^ ]*-ADV|[^ ]*-TMP|[^ ]*-EXT)([^>]*)>\^/\^\1\2\3\4 <\1\2-bN-lI-fNIL \5> <N\3-lA-f\6\7>\^/ && ($j=98)) ||
        # gerund: branch off final argument NS
        (s/\^(N)([^ ]*?)($CODA) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(NP)([^>]*)>\^/\^\1\2\3 <\1\2-bN-lI-fNIL \4> <N-lA-fNS\6>\^/ && ($j=98.5)) ||
        (s/\^(N)([^ ]*?)($CODA) (<(?:(?!VB|JJ|MD|TO|NN)[^>])*VBG[^>]*>) <(S[^ ]*-NOM)([^>]*)>\^/\^\1\2\3 <\1\2-bN-lI-fNIL \4> <N-lA-f\5\6>\^/ && ($j=99)) ||

        # branch off final modifier empty S|SBAR it-expletive trace Ne (keep around for use in cleft)
        (s/\^([VIBLAGR]-aN(?!-x)|N(?!c|-a))([^ ]*?)($CODA) (<.*) <([^>\]]*-NONE- \*EXP\*[^>\[]*)>\^/\^\1e\2\3 <\1e\2-lI-fNIL \4> <\5>\^/ && ($j=100)) ||

        #### final S
        # branch off final S-ADV with empty subject as modifier RP
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE[I]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL <I-aN-lI-f\5 \6>>\^/ && ($j=101)) ||
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE[A]S[^ ]*(?:-ADV|-PRP)[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-f\5 \6>\^/ && ($j=102)) ||
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE[I]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL <I-aN-lI-f\5 \6>>\^/ && ($j=103)) ||
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE[A]P[^ ]*(?:-ADV|-PRP)[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-f\5 \6>\^/ && ($j=104)) ||
        # branch off final S with empty subject as argument BP|IP|AP|VP
        (s/\^(N(?!-a))([^ x]*?)($CODA) (<.*-NONE-.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-k\{\6-aN}-lI-fNIL \4> <\6-aN-lN-f\5 \7>\^/ && ($j=104.9)) ||
        (s/\^(?![^ ]*-v)([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-b\{\6-aN}-lI-fNIL \4> <\6-aN-lA-f\5 \7>\^/ && ($j=105)) ||
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*-NONE-.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-b\{\6-aN}-lI-fNIL \4> <\6-aN-lA-f\5 \7>\^/ && ($j=105.1)) ||
        #(s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*-NONE-.*) <(S-TOBE([VIBA])S[^ ]*) $EMPTY_SUBJ ([^>]*)>\^/\^\1\2\3 <\1\2-b{\6-aN}-lI-fNIL \4> <\6-aN-lA-f\5 \7>\^/ && ($j=105)) ||
        # try not to take [AR]-aN-x by having [^ x]* instead of normal [^ ]*
        (s/\^(N(?!-a))([^ x]*?)($CODA) (<.*) <(S-TOBE([VIBA])P[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-k\{\6-aN}-lI-fNIL \4> <\6-aN-lN-f\5 \7>\^/ && ($j=105.9)) ||
        (s/\^(?![^ ]*-v)([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*) <(S-TOBE([VIBA])P[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-b\{\6-aN}-lI-fNIL \4> <\6-aN-lA-f\5 \7>\^/ && ($j=106)) ||
        (s/\^([SQCFVIBLAGRN])([^ x]*?)($CODA) (<.*-NONE-.*) <(S-TOBE([VIBA])P[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-b\{\6-aN}-lI-fNIL \4> <\6-aN-lA-f\5 \7>\^/ && ($j=106.1)) ||
        # branch off final 'so' + S as modifier RP
        (s/\^([SQCFVIBLAGR])([^ x]*?)($CODA) (<.*) (<[^>\]]* so\]*> <S-TOBE[V]S[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=107)) ||
        # branch off final IN|TO + S as modifier RP
        (s/\^([SQCFVIBLAGR])([^ x]*?)($CODA) (<.*) (<(?:IN)[^ ]*> <S-TOBE[V]S[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=108)) ||
        # branch off final IN|TO + S as modifier AP
        (s/\^(N)([^ ]*?)($CODA) (<.*) (<(?:IN|TO)[^>]*> <S-TOBE[V]S[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL \5>\^/ && ($j=109)) ||
        # branch off final S-ADV as modifier RP
        (s/\^([SQCFVIBLAGR])([^ x]*?)($CODA) (<.*) <(S-TOBE([IA])S[^ ]*(?:-ADV|-PRP)[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL <\6-lI-fS\5>>\^/ && ($j=110)) ||
        # branch off final S as argument VS
        (s/\^(C)([^ ]*?)($CODA) (<.*) <(S-TOBE([V])S[^>]*)>\^/\^\1\2\3 <\1\2-b\6-lI-fNIL \4> <\6-lA-f\5>\^/ && ($j=111)) ||
        # branch off ; S as conjunction (semantically viable analysis to replace bogus V-lM rules below)
        (s/\^(S|V\-iN|Q|[VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) (<.*) <[^ ]* (--|,|:|;)> (<`` ``> |<ADVP [^ ]*> )?<(S-TOBE([VIBA])S[^ ]*) ([^>]*)>\^/\^\1\2\3 <\1\2-lC-fNIL \4> <\8-c\8-fNIL <X-cX-dX \5> <\8-lC-f\7 \6\9>>\^/ && ($j=113)) ||
# SHOULD NEVER FIRE       # branch off final ADVP + S as modifier VS|IS|BS|AS
#        (s/\^(S|V\-iN|Q|[VIBLAG](?!-a))([^ ]*?)($CODA) (<.*) (<ADVP[^>]*>) <(S-TOBE([VIBA])S[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\7-lM-fNIL \5 <\7-lI-f\6>>\^/ && ($j=112)) ||
# SHOULD NEVER FIRE       # branch off final S as modifier VS|IS|BS|AS
#        (s/\^(S|V\-iN|Q|[VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) (<.*) <(S-TOBE([VIBA])S[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\6-lM-f\5>\^/ && ($j=113)) ||
        # branch off final S as argument VS|IS|BS|AS
#        (s/\^([VIBLAGR]P|NS|NP)([^ ]*?)($CODA) (<.*) <(S-TOBE([VIBA])S[^>]*)>\^/\^\1\2\3 <\1\2-b\6S-lI-fNIL \4> <\6S-lA-f\5>\^/ && ($j=114)) ||
        (s/\^(N(?!-a))([^ ]*?)(-[fghjlpi][^ \}]*) (<.*) <(S-TOBE([VIBA])[SP][^>]*)>\^/\^\1\2\3 <\1\2-k\6-lI-fNIL \4> <\6-lN-f\5>\^/ && ($j=114)) ||
        (s/\^([VIBLAGR]-aN(?!-x)|N-aD)([^ ]*?)($CODA) (<.*) <(S-TOBE([VIBA])[SP][^>]*)>\^/\^\1\2\3 <\1\2-b\6-lI-fNIL \4> <\6-lA-f\5>\^/ && ($j=114.5)) ||

        #### final SBAR
        # branch off final SBAR as modifier AP NS (nom clause):                        {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NS <NS ...> <AP <NS WH# ... t# ...>>}
        (s/\^(N)([^ ]*?)($CODA) ((?!.*<CC)<.*) <(SBAR[Q]?-NOM)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL <N-lI-f\5\6>>\^/ && ($j=115)) ||
        # branch off final SBAR as modifier Cr:                                        {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {NS <NS ...> <Cr WH# ... t# ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR$SUITABLE_REL_CLAUSE)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <C-rN-lN-f\5>\^/ && ($j=116)) ||
        # branch off final SBAR as argument NS (nom clause):                           {VP ... <SBAR [WH# what/who] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)(-[ir][^- \}]*)?($CODA) (<.*) <(SBAR[^ ]*-NOM|SBAR [^\]]*(?:when|whether))([^>]*)>\^/\^\1\2\3\4 <\1\2-bN-lI-fNIL \5> <N\3-lA-f\6\7>\^/ && ($j=117)) ||
        # branch off final SBAR as argument NS (gerund's nom clause):                  {NS ... <SBAR [WH# what/who] ... t# ...>} => {NS <NS-bNS ...> <NS ... t# ...>}
        (s/\^(N)([^ ]*?)($CODA) (<(?:(?!VB|JJ|MD|TO|NP)[^>])*VBG[^>]*>) <(SBAR[^ ]*-NOM)([^>]*)>\^/\^\1\2\3 <\1\2-bN-lI-fNIL \4> <N-lA-f\5\6>\^/ && ($j=118)) ||
#        # branch off final SBAR as argument IP:                                        {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ...]>} => {VP-bIP ... <IP ... to ...>}
#        (s/\^([VIBLAGR]-aN(?!-x)|N)([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[VP[^ ]* \[TO to\][^>]*)\]>\^/\^\1\2\3 <\1\2-b{I-aN}-lI-fNIL \4> <I-aN-lA-fNIL \6>\^/ && ($j=119)) ||
       # branch off final SBAR as modifier IP:                                        {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ...]>} => {VP-bIP ... <IP ... to ...>}
        (s/\^([VIBLAGR]-aN(?!-x)|N)([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[VP[^ ]* \[TO to\][^>]*)\]>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <I-aN-lM-fNIL \6>\^/ && ($j=119.5)) ||
#        # branch off final SBAR as modifier RP IP:                                    ----> {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ... t# ...]>} => {VP ... <IP ... to ...>}
#        s/\^([VIBLAGR]P|NS|NP)([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>\^/\^\1\2\3 <\1\2-bIP-lI-fNIL \4> <IP-lA-fNIL \6>\^/ ||
#        # branch off final SBAR as modifier RP IP:                                    {VP ... <SBAR [WH# nil] [S [NS nil t#] [ ... to ...]>} => {VP ... <RP [IP ... to ...]>}
#        s/\^([VIBLAGR]P)([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>\^/<\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP IP (from SBAR trace coindexed to empty subj)
#        s/\^(NS|NP)([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WH[^ ]*(-[0-9]+)[^ ]* \[-NONE- [^ ]*\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*T\*\5\]\] (\[\VP[^ ]* \[TO to\][^>]*)\]>\^/<\1\2\3 <\1\2-lI-fNIL \4> <AP-lI-fNIL <IP-lI-fNIL \6>>>/ ||
#        # branch off final SBAR as modifier RP (from SBAR trace coindexed to final modifier, which must have gap discharged)
#        s/\^([VIBLAGR]P)([^ ]*?)($CODA) (<.*) <(SBAR[^ ]* \[WHADVP(-[0-9]+) [^>]*) \[ADVP[^ ]* \[-NONE- \*T\*\6\]\]([\]]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <RP-lI-f\5\7>\^/ ||
        # branch off final SBAR as argument ES-gNS ('tough for X to Y' construction):  {AP ... <SBAR [WH# nil] ... for ... #t ...>} => {AP <AP-bESg ...> <ES-gNS ... for ... #t ...>}  ****SHOULD BE ES..-lN****
        (s/\^(A-aN(?!-x))([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] (\[IN for\] [^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)>\^/\^\1\2\3 <\1\2-b\{F-gN}-lI-fNIL \4> <F-gN\5-lA-fNIL \6>\^/ && ($j=120)) ||
        # branch off final SBAR as argument IP-gNS ('tough to Y' construction):        {AP ... <SBAR [WH# nil] ... to ... #t ...>} => {AP <AP-bIPg ...> <IP-gNS ... for ... #t ...>}
        (s/\^(A-aN(?!-x))([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[VP[^ ]* \[TO to\][^>]* \[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>\^/\^\1\2\3 <\1\2-b\{I-aN-gN}-lI-fNIL \4> <I-aN-gN\5-lA-fNIL \6>\^/ && ($j=121)) ||
        # branch off final SBAR as argument AP-gNS ('worth Y-ing' construction):       {AP ... <SBAR [WH# nil] ... #t ...>} => {AP <AP-bAPg ...> <AP-gNS ... #t ...>}
        (s/\^(A-aN(?!-x))([^ ]*?)($CODA) (<.*) <SBAR[^ ]* \[WHNP(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] (\[NP[^ ]* \[-NONE- \*T\*\5\]\][^>]*)\]>\^/\^\1\2\3 <\1\2-b\{A-aN-gN}-lI-fNIL \4> <A-aN-gN\5-lA-fNIL \6>\^/ && ($j=122)) ||
        # branch off final SBAR as modifier IP-gNS (NS_i [for X] to find pics of t_i): {NS ... <SBAR [WHNP# nil] ... #t ...>} => {NS <AP ...> <IP-gNS ... #t ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR[^ ]*) \[WHNP[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <I-aN-gN\6-lN-f\5 \7>\^/ && ($j=123)) ||
        # branch off final SBAR as modifier IP-gRP (NS_i [for X] to say you ... t_i):  {NS ... <SBAR [WH# nil] ... #t ...>} => {NS <AP ...> <IP-gRP ... #t ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR[^ ]*) \[WH[^ ]*(-[0-9]+) \[-NONE- 0\]\] \[S[^ ]* \[NP[^ ]* \[-NONE- \*\]\] ([^>]*\[-NONE- \*T\*\6\][^>]*)\]>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <I-aN-g\{R-aN}\6-lN-f\5 \7>\^/ && ($j=124)) ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# where/when] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/\^([SQCFVIBLAGR])([^ ]*?)($CODA) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <V-rN-lN-f\5\6>\^/ && ($j=125)) ||
        # branch off final SBAR as modifier Cr:                                        {NS ... <SBAR [WH# where/when] ... t# ...>} => {NS <VP-bNS ...> <NS ... t# ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR(?![^ ]*-NOM)[^\]]* (?:where|when))([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <C-rN-lN-f\5\6>\^/ && ($j=126)) ||
        # branch off final SBAR as modifier RC:                                        {VP ... <SBAR [WH# which] ... t# ...>} => {VP <VP ...> <RC WH# ... t# ...>}
        (s/\^([SQCFVIBLAGR])([^ ]*?)($CODA) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*which[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <V-rN-lN-f\5\6>\^/ && ($j=127)) ||
        # branch off final SBAR as modifier Cr (that|nil):                             {NS ... <SBAR [WH# which/who/that/nil] ... t# ...>} => {VP <VP ...> <Cr WH# ... t# ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <C-rN-lN-f\5\6>\^/ && ($j=128)) ||
        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-bQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
        (s/\^(N(?!-a))([^ ]*?)($CODA) (<.*) <(SBAR(?!-ADV|-TMP)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that|[^\]]*-NONE-))([^>]*)>\^/\^\1\2\3 <\1\2-k\{V-iN}-lI-fNIL \4> <V-iN-lN-f\5\6>\^/ && ($j=129)) ||
        (s/\^([VIBLAGR]-aN(?!-x)|N)([^ ]*?)($CODA) (<.*) <(SBAR(?!-ADV|-TMP)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that|[^\]]*-NONE-))([^>]*)>\^/\^\1\2\3 <\1\2-b\{V-iN}-lI-fNIL \4> <V-iN-lA-f\5\6>\^/ && ($j=129)) ||
#        s/\^(SS|VS\-iNS|QS|CS|ES|[VIBLAGR])([^ ]*?)($CODA) (<.*) <(SBAR)([^ ]* \[WH[^ ]*-[0-9]+(?![^\]]*what|[^\]]*why)[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <RC-lN-f\5\6>\^/ ||
        # branch off final SBAR as modifier RP colon:                                  {VP ... <:> <SBAR [IN because/..] ...>} => {VP <VP ...> <AP <:> <AP because ...>>}
        (s/\^([SQCFVIBLAGR])([^ ]*?)($CODA) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*<SBAR[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=130)) ||
        # branch off final SBAR as modifier AP colon:                                  {NS ... <:> <SBAR [IN because/..] ...>} => {NS <NS ...> <AP <:> <AP because ...>>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) (<[^ ]* :> <.*<SBAR[^>]*)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL \5>\^/ && ($j=131)) ||
        # branch off final SBAR as modifier ES:                                        {NS ... <SBAR [IN because/..] ...>} => {NS <NS ...> <AP ...>}
        (s/\^([SQCFVIBLAGRN])([^ ]*?)($CODA) (<.*) <(SBAR[^ ]* \[[^ ]* for\] [^>]*)>\^/\^\1\2\3 <\1\2-bF-lI-fNIL \4> <F-lA-f\5>\^/ && ($j=132)) ||
        # branch off final SBAR as modifier RP:                                        {VP ... <SBAR [IN because/..] ...>} => {VP <VP ...> <AP ...>}
#WORSE:        s/\^(SS|VS\-iNS|QS|CS|ES|[VIBLAGR])([^ ]*?)($CODA) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*-TMP)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <RP-t-lM-f\5\6>\^/ ||
        (s/\^([SQCFVIBLAGR])([^ ]*?)($CODA) (<.*) <(?![^ ]*-PRD)(SBAR[^ ]*(?:-ADV|-LOC|-TMP|-CLR)|SBAR[^ ]*(?: \[[^ ]* [^ ]*\]| \[[^ ]* \[[^ ]* [^ ]*\]\])* \[IN (?!that))([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-f\5\6>\^/ && ($j=133)) ||
        # branch off final SBAR as modifier AP:                                        {NS ... <SBAR [IN because/..] ...>} => {NS <NS ...> <AP ...>}
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SBAR[Q]?-LOC|SBAR[Q]?-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-f\5\6>\^/ && ($j=134)) ||
        # branch off final SBAR as argument AP NS (nom clause following 'being'):      {NS ... being ... <SBAR [WH# what/who] ... t# ...>} => {NS <NS-bAP ... being ...> <AP <NS ... t# ...>>}
        (s/\^(N)([^ ]*?)($CODA) (.* being>.*) <(SBAR-NOM|SBAR-PRD)([^>]*)>\^/\^\1\2\3 <\1\2-b\{A-aN}-lI-fNIL \4> <A-aN-lA-fNIL <N-lI-f\5\6>>\^/ && ($j=135)) ||
#        # branch off final SBAR as argument QC:                                        {XP ... <SBAR [WH# what/whichN/who/where/why/when/how/if/whether] ... t# ...>} => {XP <XP-bQC ...> <QC ... t# ...>}   ****SHOULD BE QC****
#        s/\^([VIBLAGR]P|NS|NP)([^ ]*?)($CODA) (<.*) <(SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)>\^/\^\1\2\3 <\1\2-bVS-iNS-lI-fNIL \4> <VS-iNS-lA-f\5\6>\^/ ||
        # delete final empty SBAR given gap Cr and re-run:
        (s/\^([SQCFVIBLAGRN])(-h\{C-rN}|-h\{I-aN})(-[0-9]+)([^ ]*) (<.*) <SBAR[^>\]]*\[-NONE- \*(?:T|ICH)\*\3\][^>\[]*>\^/<\1\2\3\4 \5>/ && ($j=136)) ||
        # branch off final SBAR as argument CS:                                        {XP ... <SBAR [IN that/nil] ...>} => {XP <XP-bCS ...> <CS ...>}   *****SHOULD BE CS******
        (s/\^(N(?!-a))([^ ]*?)($CODA) (<.*) <(SBAR)([^>]*)>\^/\^\1\2\3 <\1\2-kC-lI-fNIL \4> <C-lN-f\5\6>\^/ && ($j=136.9)) ||
        (s/\^([VIBLAGRN]-a[ND](?!-x)|N(?!-a))([^ ]*?)($CODA) (<.*) <(SBAR)([^>]*)>\^/\^\1\2\3 <\1\2-bC-lI-fNIL \4> <C-lA-f\5\6>\^/ && ($j=137)) ||

        #### final RP
        # branch off final modifier RP colon
        (s/\^([SQCFVIBLAGR](?![^ ]*-x))([^ ]*?)($CODA) ((?!<[^ ]*-ADV[^>]*> <[^ ]* :>)<.*) (<[^ ]* :> <.*)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=138)) ||
        # branch off final modifier RP
        (s/\^([SQCFVIBLAGR](?![^ ]*-x))([^ ]*?)($CODA) (<.*) <(?![^ ]*-PRD)(NP-ADV|NP-TMP|NP-EXT)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNS-TMP\6>\^/ && ($j=138.5)) ||
        (s/\^([SQCFVIBLAGR](?![^ ]*-x))([^ ]*?)($CODA) (<.*) <(?![^ ]*-PRD)(RB|ADVP|PP|UCP|FRAG|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-f\5\6>\^/ && ($j=139)) ||
        # branch off final SQ|SINV as argument SS
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<.*) <(SQ|SINV)([^>]*)>\^/\^\1\2\3 <\1\2-bS-lI-fNIL \4> <S-lA-f\5\6>\^/ && ($j=140)) ||
        # branch off final INJP as argument CS
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<.*) <(INTJ)((?![^ ]*-ADV)[^>]*)>\^/\^\1\2\3 <\1\2-bC-lI-fNIL \4> <C-lA-f\5\6>\^/ && ($j=141)) ||
        # branch off final argument LS ('had I only known' construction)
        (s/\^([VIBLAGR]-aN(?!-x))([^ ]*?)($CODA) (<VBD had>) (.*<NP.*<VP.*)\^/\^\1\2\3 <\1\2-bL-lI-fNIL \4> <L-lA-fNIL \5>\^/ && ($j=142)) ||

        #### final AP
#        # semi/dash splice
#        s/\^(NS)([^ ]*?)($CODA) (<.*) (<[^ ]* ;>) (<.*)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <C\1 \5 <\1\2-lI-fNIL \6>>\^/ ||
        # gerund: delete initial empty subject (passing -o to head)
        (s/\^(N(?!-a))([^ ]*?)($CODA) <NP-SBJ[^ ]* \[-NONE- [^>]*> (<.*)\^/<\1\2\3 \4>/ && ($j=143)) ||
        # branch off final modifier AP appositive NS (passing -o to head)
        # This rule match beyond the current constituent!!! It looks for <NP...> ... <N[PS]...> where the 1-char cat may have the second term as N or N-aN. Try N[PS]? but other sent failed b/c this rule hit wrongly
        (s/\^(N)([^ ]*?)($CODA) ((?=.*<(?:NP|[^ ]*-NOM).*<(?:N[PS]?(?!-[pc])|[^ ]*-NOM))(?!.*<CC)<.*) <(NP|[^ ]*-NOM)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL <N-lI-fNS\6>>\^/ && ($j=144)) ||

        # branch off final modifier AP infinitive phrase (with TO before any VBs) (passing -o to head)  ****************
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(VP(?:(?!\[VB)[^>])*\[TO[^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-fNIL <I-aN-lI-f\5>>\^/ && ($j=145)) ||
        # branch off final modifier AP (passing -o to head)
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(RRC|PP|ADJP|ADVP|RB|UCP|[^ ]*-LOC|[^ ]*-TMP|[^ ]*-EXT|SBAR[^ ]* \[IN (?!that))([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-f\5\6>\^/ && ($j=146)) ||
        # branch off final modifier AP (from VP) (passing -o to head)
        (s/\^(N)([^ ]*?)($CODA) (.*<(?:NP|NN).*) <(VP)([^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <A-aN-lM-f\5\6>\^/ && ($j=147)) ||
        # branch off final argument LP (passing -o to head)
        (s/\^(N)([^ ]*?)($CODA) (.* having>.*) <(VP)([^>]*)>\^/\^\1\2\3 <\1\2-b\{L-aN}-lI-fNIL \4> <L-aN-lA-f\5\6>\^/ && ($j=148)) ||
        # branch off final argument VS\-iNS (passing -o to head)  **WH VS\-iNS WILL NEVER FIRE**
        (s/\^(N)([^ ]*?)($CODA) (<.*) <(SQ(?=[^ ]* \[WH))([^>]*)>\^/\^\1\2\3 <\1\2-bV-iN-lI-fNIL \4> <V-iN-lA-f\5\6>\^/ && ($j=149)) ||

        #### final misc needed by SS   *****COULD BE GENERALIZED******
        # branch off final 'so' + S
        (s/\^([SQCFVIBLAG](?!-a))([^ ]*?)($CODA) (<.*) (<[^ ]* so> <S(?![A-Z])[^>]*>)\^/\^\1\2\3 <\1\2-lI-fNIL \4> <R-aN-lM-fNIL \5>\^/ && ($j=150)) ||

        ######################################################################
        ## 5. LOW BRANCHING / HIGH PRECEDENCE INITIAL CONSTITUENTS

        #### VS
        # inverted declarative sentence: branch off final subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) (<.*) <NP([^ ]*-SBJ)([^>]*)>\^/\^\1\2\3 <\1-aN\2-lI-fNIL \4> <N-lA-fNS\5\6>\^/ && ($j=151)) ||
        # inverted declarative sentence: branch off final cleft subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) (<.*) <NP([^ ]*-CLF)([^>]*)>\^/\^\1\2\3 <\1-aNc\2-lI-fNIL \4> <Nc-lA-fNS\5\6>\^/ && ($j=151)) ||
        # [VIBLAG] sentence: branch off initial CS subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <(SBAR[^ ]*-SBJ[^\]]*that)([^>]*)> (<.*)\^/\^\1\2\3 <C-lA-f\4\5> <\1-aN\2-lI-fNIL \6>\^/ && ($j=152)) ||
#        # [VIBLAG] sentence: branch off initial VS\-iNS subject
#        s/\^([VIBLA])S([^ ]*?)($CODA) <(SBAR(?!-ADV)[^\]]*(?:whether|if)|SBAR[^ ]* \[WH(?![^\]]*that))([^>]*)> (<.*)\^/\^\1S\2\3 <VS-iNS-lA-f\4\5> <\1P\2-lI-fNIL \6>\^/ ||
        # [VIBLAG] sentence: branch off initial ES subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <(SBAR[^ ]*-SBJ[^\]]*for)([^>]*)> (<.*)\^/\^\1\2\3 <F-lA-f\4\5> <\1-aN\2-lI-fNIL \6>\^/ && ($j=153)) ||
        # [VIBLAG] sentence: branch off initial IP subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <(SheadI[^ ]*-SBJ)([^>]*)> (<.*)\^/\^\1\2\3 <I-aN-lA-f\4\5> <\1-aN\2-lI-fNIL \6>\^/ && ($j=154)) ||
        # [VIBLAG] sentence: branch off initial NS subject
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <(NP)(-CLF)([^>]*)> (<.*)\^/\^\1\2\3 <Nc-lA-f\4\5\6> <\1-aNc\2-lI-fNIL \7>\^/ && ($j=154.5)) ||
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <(NP)(-SBJ)?([^>]*)> (<.*)\^/\^\1\2\3 <N-lA-f\4\5\6> <\1-aN\2-lI-fNIL \7>\^/ && ($j=154.5)) ||
        (s/\^([VIBLAG](?!-aN(?!e)))([^ ]*?)($CODA) <([^ ]*-NOM|[^ ]*-SBJ)([^>]*)> (<.*)\^/\^\1\2\3 <N-lA-f\4\5> <\1-aN\2-lI-fNIL \6>\^/ && ($j=155)) ||

        #### NS
        # branch off initial punctuation
        (s/\^(N(?!-[pc]))([^ ]*?)($CODA) <([^ ]*) ($INIT_PUNCT)> (<.*)\^/\^\1\2\3 <\4-lM-f\4 \5> <\1\2-lI-fNIL \6>\^/ && ($j=156)) ||
        # branch off initial parenthetical sentence with extraction
        (s/\^(N)([^ ]*?)($CODA) <(PRN|S(?![A-Z]))([^>]* \[S[^>]*\[-NONE- \*INTERNAL\*(-[0-9]+)[^>]*)> (<.*)\^/\^\1\2\3 <V-gS\6-lN-f\4\5> <\1\2-lI-fNIL \7>\^/ && ($j=157)) ||
        # branch off initial pre-determiner
        (s/\^(N(?!-aD))([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(PDT|RB)([^>]*)> (<.*)\^/\^\1\2\3\4\5 <D-lA-f\6\7> <N\2\3\4-lI-fNIL \8>\^/ && ($j=158)) ||
        # branch off initial determiner (wh)
        (s/\^(N(?!-aD))([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(WDT|WP\$)([^>]*)> (<.*)\^/\^\1\2\3\4\5 <D\3-lA-f\6\7> <N-aD\2\4-lI-fNIL \8>\^/ && ($j=159)) ||
        # branch off initial determiner (non-wh)
        (s/\^(N(?!-aD))([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <NP([^>]*\[POS 's?\])([^>]*)> (<.*)\^/\^\1\2\3\4\5 <D-lA-fNS\6\7> <N-aD\2\3\4-lI-fNIL \8>\^/ && ($j=159.5)) ||
        (s/\^(N(?!-aD))([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(DT)([^>]*)> (<.*)\^/\^\1\2\3\4\5 <N-b\{N-aD}-lI-f\6\7> <N-aD\2\3\4-lA-fNIL \8>\^/ && ($j=159.7)) ||
        (s/\^(N(?!-aD))([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(DT|PRP\$|PRP|NP[^>]*\[POS 's?\])([^>]*)> (<.*)\^/\^\1\2\3\4\5 <D-lA-f\6\7> <N-aD\2\3\4-lI-fNIL \8>\^/ && ($j=160)) ||
        # branch off initial modifier A-aN-x (wh adv)
        (s/\^N(-aD)?([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(WHADJP|WRB)([^>]*)> (.*<(?:DT|NP|NX|NN|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^N\1\2\3\4\5 <A-aN-x\3-lM-f\6\7> <N-aD\2\4-lI-fNIL \8>\^/ && ($j=161)) ||
        # branch off initial modifier A-aN-x (noun modifier)
        (s/\^N(-aD)?([^ ]*?)(-[ir][^- \}]*)?([^ \}]*?)($CODA) <(NN[^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^N\1\2\3\4\5 <A-aN-x-lM-fNIL <N-aD-lI-f\6>> <N-aD\3\4-lI-fNIL \7>\^/ && ($j=161.2)) ||
        # branch off initial modifier A-aN-x
        (s/\^(N-aD)([^ ]*?)(-[fghjlrp][^ \}]*?)N-aD <(NN)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^\1\2\3N <A-aN-x-lM-f\4\5> <\1\2-lI-fNIL \6>\^/ && ($j=161.39)) ||
        (s/\^(N)([^ ]*?)(-[fghjlrp][^ \}]*?)N-aD <(NN)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^\1\2\3N <A-aN-x-lM-f\4\5> <\1-aD\2-lI-fNIL \6>\^/ && ($j=161.4)) ||
        (s/\^N(-aD)?([^ ]*?)((?:-[fghjlrp0-9][^ \{\}]*|-[fghjlrp]\{[^ \{\}]*\})*) <(NN)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^N\1\2\3 <A-aN-x-lM-f\4\5> <N-aD\2-lI-fNIL \6>\^/ && ($j=161.5)) ||
        (s/\^N(-aD)?([^ ]*?)((?:-[fghjlrp0-9][^ \{\}]*|-[fghjlrp0-9]\{[^ \{\}]*\})*) <(CD|QP|JJ|ADJP|WHADJP|IN|PP|RB|TO|ADVP|VB|UCP|[^ ]*-ADV|[^ ]*-LOC|[^ ]*-TMP|SBAR[^ ]* \[IN (?!that))([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^N\1\2\3 <A-aN-x-lM-f\4\5> <N-aD\2-lI-fNIL \6>\^/ && ($j=162)) ||
        (s/\^N(-aD)?([^ ]*?)($CODA) <(NAC)([^>]*)> (.*<(?:DT|NN|NX|NP|VB|VP|JJ|ADJP|CD|\$|QP \[\$).*)\^/\^N\1\2\3 <A-aN-x-lM-fNIL <N-lI-f\4\5>> <N-aD\2-lI-fNIL \6>\^/ && ($j=163)) ||
        # rebinarize QP containing dollar sign followed by *U*, and continue
        (s/\^(N|A-aN-x)([^ ]*?)($CODA) <QP ([^>]*)\[([^ ]* [^ ]*[\$\#][^ ]*)\] ([^>]*)>(?: <-NONE- \*U\*>)?\^/<\1\2\3 \4\[\5\] \[QP \6\]>/ && ($j=164)) ||
        # branch off currency unit followed by non-final *U*
        (s/\^(N|A-aN-x)([^ ]*?)($CODA) (<[^ ]* [\$\#][^ ]*>.*) <-NONE- \*U\*> (<.*)\^/\^\1\2\3 <A-aN-x-lM-fNIL \4> <N-aD-lI-fNIL \5>\^/ && ($j=165)) ||
        # rebinarize currency unit followed by QP
        (s/\^(N|A-aN-x)([^ ]*?)($CODA) <([^ ]* [\$\#][^ ]*)> (.*?)( <-NONE- \*U\*>)?\^/\^\1\2\3 <\$-lI-f\4> <A-aN-x-lM-fNIL \5>\^/ && ($j=166)) ||

        #### U
        # branch off initial U
        (s/\^(U)([^ ]*?)($CODA) <([^ ]*) ([^<>]*)> (<.*)\^/\^\1\2\3 <\1-lI-f\4 \5> <\1\2-lI-fNIL \6>\^/ && ($j=166.5)) ||
        #(s/\^(U)([^ ]*?)($CODA) (<.*) <(^>]*)>\^/\^\1\2\3 <\1\2-lI-fNIL \4> <\1-lI-fNIL \5>\^/ && ($j=166.5)) ||

        #### A-aN-x|R-aN-x|NP
        # branch off initial modifier R-aN-x
        (s/\^(A-aN-x)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <(W[^>]*)> (<.*)\^/\^\1\2\3\4 <R-aN-x\3-lM-f\5> <A-aN-x-lI-fNIL \6>\^/ && ($j=167)) ||
        (s/\^(A-aN-x)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <NN([^>]*)> (<.*)\^/\^\1\2\3\4 <R-aN-x-lM-fNN\5> <A-aN-x\3-lI-fNIL \6>\^/ && ($j=167.5)) ||
        (s/\^(A-aN-x)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <([^>]*)> (<.*)\^/\^\1\2\3\4 <R-aN-x-lM-f\5> <A-aN-x\3-lI-fNIL \6>\^/ && ($j=168)) ||
#        s/\^(A-aN-x)([^ ]*) <([^>]*)> (<.*)\^/\^\1\2 <R-aN-x-lM-f\3> <A-aN-x-lI-fNIL \4>\^/ ||
        (s/\^(R-aN-x)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <(W[^>]*)> (<.*)\^/\^\1\2\3\4 <R-aN-x\3-lM-f\5> <R-aN-x-lI-fNIL \6>\^/ && ($j=169)) ||
        (s/\^(R-aN-x)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <([^>]*)> (<.*)\^/\^\1\2\3\4 <R-aN-x-lM-f\5> <R-aN-x\3-lI-fNIL \6>\^/ && ($j=170)) ||
#        s/\^(R-aN-x)([^ ]*) <([^>]*)> (<.*)\^/\^\1\2 <R-aN-x-lM-f\3> <R-aN-x-lI-fNIL \4>\^/ ||
        (s/\^(N-aD)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <(W[^>]*)> (<.*)\^/\^\1\2\3\4 <A-aN-x\3-lM-f\5> <N-aD-lI-fNIL \6>\^/ && ($j=171)) ||
        (s/\^(N-aD)([^ ]*?)(-[ir][^- \}]*)?([^ ]*) <([^>]*)> (<.*)\^/\^\1\2\3\4 <A-aN-x-lM-f\5> <N-aD\3-lI-fNIL \6>\^/ && ($j=172)) ||
#        s/\^(N-aD)([^ ]*) <([^>]*)> (<.*)\^/\^\1\2 <A-aN-x-lM-f\3> <NP-lI-fNIL \4>\^/ ||
        (s/\^(X-cX-dX)([^ ]*) <([^>]*)> (<.*)\^/\^\1\2 <\1-lM-f\3> <\1-lI-fNIL \4>\^/ && ($j=173)) ||
#        (s/\^(X-cX-dX)([^ ]*) ([^<>]*)\^/\^X-cX-dX\2 \3\^/ && ($j=174)) ||

		######################################################################
        ## 0. TURN {NS..<NP..>} TO {NS..<NS..>}     AND      {NS.. <DT..> <NN..>} TO {NS.. <DT..> <NP..>}  (this is to fix sent 2921)
#        (s/\^NS([^ ]*) <NP([^>]*)>([ ]*)\^/\^NS\1 <NS\2>\3\^/ && ($j=0.5)) ||
#        (s/\^NS([^ ]*) <DT([^>]*)> <NN([^>]*)>([ ]*)\^/\^NS\1 <DT\2> <NP\3>\4\^/ && ($j=0.6)) ||
        ## {DD.. <NP..> ..} TO {DD.. <NS..> ..}  (this is to fix sent 7419)
#        (s/\^DD([^ ]*) <NP(.*)\^/\^DD\1 <NS\2\^/ && ($j=0.7)) ||
#		(s/<NP /<NS /g && ($j=0.1)) ||
#		(s/<NN /<NP /g && ($j=0.2)) ||
        
        #### panic as MWE
#        s/\^([VIBLAGR]P|NS|NP)([^ ]*?)($CODA) (<.*) <(?!-NONE-)([^ ]*) (..?.?.?)>\^/\^\1\2\3 <\1\2-bAP\6-fNIL \4> <AP\6-lA-f\5 \6>\^/ ||

        1 ) {
           	if ($j==0) {
           		while( s/\^(.*)<NP(.*)\^/\^\1<N\2\^/ && ($j=0.1) ) {}  #(this is to fix sent 12157)
        		while( s/\^(.*)<NN(S|P)? (.*)\^/\^\1<NP\2 \3\^/ && ($j=0.2) ) {}  #(this is to fix sent 7919)
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

   	$k = 0;
    while (
	   debug($step, "?? $_") ||

           #### -g/h/i (gap/trace tags)
           # propagate -g gap tag from parent to each child dominating trace with matching index
           (s/\^([^ \{]*-iN)(-g\{[VC]-rN})(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjlpi][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)\^/\^\1\2\3\4 <\5\2\3\6>\7\^/  && ($k=0.5)) ||
           (s/\^([^ \{]*-iN)(-g[^- ]*)(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjlpi][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)\^/\^\1\2\3\4 <\5\2\3\6>\7\^/  && ($k=1)) ||
           (s/\^([^ \{]*)(-g\{[VC]-rN})(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjl][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)\^/\^\1\2\3\4 <\5\2\3\6>\7\^/  && ($k=1.5)) ||
           (s/\^([^ ]*)(-g[^- \{\}]*|-g\{[^ \{\}]*\})(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjl][^>]*\[-NONE- *\*(?:T|ICH|INTERNAL)\*\3\][^>]*)>(.*)\^/\^\1\2\3\4 <\5\2\3\6>\7\^/  && ($k=2)) ||
           # propagate -h right-node-raising gap tag from parent to each child dominating trace with matching index
           (s/\^([^ ]*)(-h[^- ]*|-h\{[VC]-rN}|-h\{R-aN}|-h\{I-aN})(-[0-9]+)((?![0-9]).*) <((?![^ ]*\2\3)[^ ]*?)(-[fghjl][^>]*\[-NONE- *\*(?:RNR|ICH)\*\3\][^>]*)>(.*)\^/\^\1\2\3\4 <\5\2\3\6>\7\^/  && ($k=3)) ||
           # propagate -v passivization tag from parent to first child containing a trace
           (s/\^([^ ]*)(-v[^- \}]*)((?:.(?!-NONE-))*) <((?![^ ]*\2)[^ ]*?)(-[fghjl][^>]*\[-NONE- \*(?:-[0-9]*)?[\*\]]*\])(.*)\^/\^\1\2\3 <\4\2\5\6\^/  && ($k=4)) ||
#           # add -j it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
#           (s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-j)([^ ]*?)(-[fghjlp][^>]*\[[^ ]*\2[^>]*)>/\1<\3-jNe\4>/  && ($k=4)) ||
#           # add -a spec tag to sibling of cleft complement following expletive trace (-a then passed down thru syn heads)
#           (s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-aNe)([^ ]*?)(?:-a[A-Z]+)?(-[bfghjlp][^>]*)> <([^ ]*\2[^>]*)>/\1<\3-aNe\4> <\5>/  && ($k=5)) ||
#           # when attaching argument preceded by coindexed expletive, subst -aN for -aNe
           (s/\^(Ne.*)\^ <([A-Z]*)-aN([^e][^>]*)>/\^\1\^ <\2-aNe\3>/ && ($k=6)) ||
#           # add -a spec tag to sibling of cleft complement containing expletive trace (-a then passed down thru syn heads)
#           (s/<(?![^ ]*-aNe)([^ ]*?)(?:-a[A-Z]+)?(-[bfghjlp][^>]*-NONE- \*EXP\*(-[0-9]+)[^0-9][^>]*)> <([^ ]*\3[^>]*)>/<\1-aNe-IDIDTHIS\2> <\4>/  && ($k=6)) ||
#           # turn last -i tag into -a
#           s/\^([^ ]*)-i(?!.*<[^ ]*-i.*\})(.*)\^/\^\1-a\2\^/ ||
##           # add -i it-cleft tag to each constituent following expletive trace that non-immediately dominates cleft complement with matching index
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)\{(?![^ ]*-i)([^ ]*?)(-[fghjlp].*\[[^ ]*\2.*)\\^/\1\{\3-iNSe\4\\^/ ||
##           # turn last -i tag into -a
##           # add -a spec tag to each constituent following expletive trace that immediately dominates cleft complement with matching index (-a then passed down thru syn heads)
##           s/(-NONE- \*EXP\*(-[0-9]+)[^0-9].*)<(?![^ ]*-a)([^ ]*?)(-[fghjlp].*\[[^ ]*\2.*)>/\1<\3-aNSe\4>/ ||
           0 ) {
           	debug($step, " used rule k=$k")
           }

    ####################
    ## mark current as external...
    s/\^(.*?)\^/\(\1\)/;
    ## mark first unexpanded child as current...
    s/<(.*?)>/\^\1\^/;
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
         # empty category deletion (double)
         s/ \([^ ]*-v[^ ]*[^ ]* \([^ ]*-NONE- \*(?:-[0-9]*)?\*?[^\)]*\)\)// ||
         # empty category deletion (single)
         s/ \([^ ]*-v[^ ]*[^ ]*-NONE- \*(?:-[0-9]*)?\*?[^\)]*\)// ||
         0 ) {}

  # remove trace numbers from gaps
  s/(-[gh]\{[VC]-rN})-[0-9]+/\1/g;
  s/(-[gh]\{?[A-Z](-a.)?}?)-[0-9]+/\1/g;
#  # turn right-node-raising gaps into complement tags
#  s/-h([A-Z]+)-[0-9]+/-b\1/g;
#  # turn expletive it-cleft traces into specifier tags
#  s/(-a[A-Z]+)?-j([A-Z]+)/-a\2/g;

  # correct VBNs
  s/\(L-aN([^ ]*)-fVB[A-Z]*/\(L-aN\1-fVBN/g;
  s/\(([AR]-aN$CODA-fVBN) ([^ \(\)]*)\)/\(\1 \(L-aN-bN-fVBN \2\)\)/g;   ## mark passive
#  s/\((A-aN|A-aN-x|R-aN|R-aN-x)([^ ]*)-fVB[ND]/\(\1\2-v-fVBN-v/g;     ## mark passive
  s/\((A-aN|A-aN-x|R-aN|R-aN-x|N-aD|N)([^ ]*)-fVB[G]/\(G-aN\2-fVBG-o/g;   ## mark progressive/gerund

  # change $ POS to N
  s/\(\$/\(N/g;

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

  # convert added extraposition '-k' into '-h'
  s/\(([^ ]*)-k([^ ]*) ([^\(\)]*)\)/\(\1-h\2 \(\1-b\2 \3\)\)/g;
  s/-k(?=[^ \)]* [^\(])/-b/g;
  while ( s/-k([^ \)]*) /-h\1 / ) { }

  # add unary branch to generalize gap categories
  s/\(([^ ]*)(-l[^I])([^ ]*) ([^\(\)]*)\)/\(\1\2\3 \(\1-lI\3 \4\)\)/g;
#  s/\(([^ ]*)-g(N|S)([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1-u\2\3 \4\)\)/g;
  s/\(([^ ]*)-g(\{R-aN}|\{C-rN})(?!})([^ ]*) ([^\(\)]*)\)/\(\1-g\2\3 \(\1\3 \4\)\)/g;

  # elim -lI (default)...
  s/-lI//g;

  # output
  print $_;
}
