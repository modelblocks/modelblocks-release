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

getopts('debug,psg', \ my %opts );

$DEBUG = 0;
if ($opts{d}) {
  $DEBUG = 1;
}

$PSG = 0;
if ($opts{p}) {
  print STDERR "Running psg mode\n";
  $PSG = 1;
} else {
  print STDERR "Running non-psg mode\n";
}

sub debug {
  if ($DEBUG) {
    $msg = $_[1];
    print stderr $_[0] , " " , $msg, "\n";
  }
}

# luan: commented out to simplify b/c now model role as '-l' tag
#$SRL = "\w+\!ldelim\!";

## for each tree...
while ( <> ) {

  #### normalize spacing to de-clutter rules
  # never more than one space
  s/ +/ /g;
  # no spaces before close parens
  s/ +\)/\)/g;

  ## misc empty category madness...
  # line 640 (incl 01):
  s/ADVP\|PRT/ADVP/;
  # line 16932 (incl 01):
  s/PRT\|ADVP/ADVP/;

  # line 722&al: some annotator is a fan of matt johnson
  s/\(DT the\) \(DT the\)/\(DT the\)/;
  # line 1187&al: sentence segmenter error
  s/\(NP \(DT the\)\)\)/\(NP \(DT the\) \(NNP U.S.\)\)\)/;
  # line 1574: tokenizer
  s/\(RB close\) \(NP/\(VB close\) \(NP/;
  # line 3780:
  s/\(VP \(NNP Face/\(VP \(VBP Face/;
#  #line 4371:
#  s/\(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBG alternating\) \(NP \(NP \(NNS feats\)\)/\(NP \(NP \(VBG alternating\) \(NP \(NP \(NNS feats\)\)/;
  # line 5393:
  s/\(VP \(POS 's\)/\(VP \(VBZ 's\)/;
  # line 5823:
  s/\(IN for\) \(S-NOM \(NP-SBJ \(NNP Congress\)\) \(VP \(TO to\)/\(IN for\) \(S \(NP-SBJ \(NNP Congress\)\) \(VP \(TO to\)/;
  # line 7440:
  s/\(VP \(JJ own\)/\(VP \(VBP own\)/;
  # line 8535:
  s/\(VP \(NNP Let\)/\(VP \(VB Let\)/;
  # line 8803:
  s/\(VP \(NNP Meet\)/\(VP \(VB Meet\)/;
  # line 8935:
  s/\(VP \(JJ oversold\)/\(VP \(VBN oversold\)/;
  # line 11592:
  s/\(IN as\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBZ is\)/\(IN as\) \(S \(NP-SBJ \(-NONE- \*\)\) \(VP \(VBZ is\)/;
  # line 13002: conj scoping error
  s/\(VP \(VBD agreed\) \(PP-CLR \(IN on\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \(VP (\(VP \(VBG denying\) \(NP \(DT the\) \(NN injunction\)\)\)) \(CC and\) (\(VP \(VBD did\).*work\))\)\)\)/\(VP \(VP \(VBD agreed\) \(PP-CLR \(IN on\) \(S-NOM \(NP-SBJ \(-NONE- \*\)\) \1\)\)\) \(CC and\) \2/;
  # line 17656:
  s/\(PP-CLR \(TO to\) \(S-NOM \(NP-SBJ \(NNP Helmuth\)/\(PP-CLR \(TO to\) \(S \(NP-SBJ \(NNP Helmuth\)/;
  # line 6440&al: S-NOM applied too liberally
  s/(\(PP[^ ]* \((?:IN|RB) [Aa]s\)) \(S-NOM /\1 \(S /g;
  s/(\(PP[^ ]* \((?:IN|RB) (?:[Ww]ith|[Bb]efore|[Aa]fter|[Ll]ike|[Uu]ntil|[Ff]or)\)) \(S-NOM (?!(\([^\(\)]*\([^\(\)]*\)\) )?\(NP-SBJ[^ ]* \(-NONE-|\(NP-SBJ[^\)]*California|\(NP-SBJ[^\)]*B\.A\.T|\(NP-SBJ[^\)]*things)/\1 \(S /g;
  # line 17057:
  s/\(VP \(JJ unhindered\)/\(VP \(VBN unhindered\)/;
  # line 32010:
  ### FAILED: s/\(S (\(\S \(NP-SBJ \(-NONE- \*\)\) \(VP \(TO to\) \(VP \(VB press\) \(NP \(DT a\) \(JJ strict\) \(JJ seven-point\))/\(S-4 \1/; 
  # line 34322:
  # line 38511:
  
  # line 20481:
  s/\(S-NOM (\(NP-SBJ[^ ]* \(-NONE- \*\)\) \(VP (\(ADVP \(RB just\)\) )?\(VBN? )/\(S \1/;

  # line 10 (incl 01):
  s/(\(WHADVP-3 \(-NONE- 0\) \) \(S  \(NP-SBJ \(-NONE- )\*(\) \) \(RB not\)  \(VP \(TO to\)  .*) \(ADVP \(-NONE- \*T\*-3\) \)/\1\*T\*-3\2/;
  # line 1281 (incl 01):
  s/\(NP \(NNP American\) \(NNP City\) \)\)\) \(NP-TMP \(-NONE- \*T\*-1\) \)/\(NP \(NNP American\) \(NNP City\) \)\)\)/;
  # line 14650 NOTE: this is an annotation bug, but it seems to hurt not help
  #s/\(DT a\) \(WDT which/\(WDT which/;
  # line 15834,22399 (incl 01):  NOTE: this is not a bug, but a reannotation!
  s/\(WHNP-2 *\(-NONE- *0\) *\) *\(IN *as\)(.*)\(S *\(NP-SBJ *\(-NONE- *\*-1\) *\) *\(NP-PRD *\(-NONE- *\*T\*-2\) *\) *\)/\(WHNP-2 \(IN as\)\)\1\(NP \(-NONE- \*\)\) \(NP \(-NONE- \*T\*-2\)\)/;
  # line 16659 (02to21):
  s/\(WHADVP-2 \(WRB how\) \(NP \(NP \(JJ much\)\)/\(WHNP-2 \(WHNP \(WHNP \(WRB how\) \(JJ much\)\)/;
  # line 19318 (incl 01):  NOTE: this is not a bug, but a reannotation!
  s/\(SBARQ *\(WHADVP *\(WRB *why\) *\) *\(FRAG *\(SQ *\(-NONE- *\*\?\*\) *\) *\) *\)/\(SBARQ \(WRB why\)\)/;
  # line 4914,27175 (incl 01):
  s/\(SBAR *\(WHNP-5 *\(-NONE- *0\) *\) *\(S(?:-PRP)? *\(-NONE- *\*ICH\*-4\) *\) *\)(?: *\(NP *\(-NONE- *\*ICH\*-1\) *\))?(.*)\(-NONE- *\*T\*-5\)/\1\(-NONE- \*-5\)/;
  # line 20028 (incl 01):
  s/\(VBN *called\) *\(S *\(NP-SBJ *\(-NONE- *\*-2\) *\) *\(NP-PRD *\(-NONE- *\*\?\*\) *\) *\)/\(VBN called\) \(NP \(-NONE- 0\)\)/;
  # line 21405 (incl 01):
  s/\(SBAR *\(WHADVP *\(-NONE- *0\) *\) *\(S *\(-NONE- *\*T\*-1\) *\) *\)/\(S \(-NONE- \*T\*-1\)\)/;
  # line 23454 (02to21):
  s/\(PP \(IN of\) \(W?H?NP \(W/\(WHPP \(IN of)\ \(WHNP \(W/;
  # line 30323 (incl 01):
  s/\(WHNP-2  \(WHNP \(-NONE- 0\) \) \(SBAR \(-NONE- \*ICH\*-4\) \)\)/\(WHNP-2 \(-NONE- 0\)\)/;
  # line 31306 (incl 01):
  s/\(VBN nicknamed\)  \(S  \(NP-SBJ \(-NONE- \*-1\) \) \(NP-PRD \(-NONE- \*ICH\*-2\) \)\)/\(VBN nicknamed\) \(NP \(-NONE- 0\)\)/;
  # line 39118 (incl 01):
  s/\(SBAR *\(WHADVP-1 *\(-NONE- *0\) *\) *\(IN *for\)(?!.*\*T\*-1)/\(SBAR \(IN for\)/;
  # line 34404 (incl 01):
  s/(\(DT The\) \(NN sense\))  \(SBAR \(-NONE- \*EXP\*-1\) \)/\1/;
  # line 24441 (incl 01):
  s/\(PP *\(NP *\(VBG/\(PP \(VP \(VBG/;
  # line 32473 (02to21):
  s/\(WRB how\) \(RB much\)/\(WRB how\) \(JJ much\)/;
  # line 39525 (incl 01):
  s/\(NP *\(VB buy\) *\(CC and\) *\(VB sell\) *\)/\(VB \(VB buy\) \(CC and\) \(VB sell\)\)/;

  ## trace errors...
  # line 962:
#  s/\(SBAR-PRD \(WHNP \(WP what\)\)/!!!!!!!FOUND!!!!!!!/;
  s/\(SBAR-PRD \(WHNP \(WP what\)\) \(S \(NP-SBJ-3 \(PRP you\)\) \(VP \(VBP 're\) \(VP \(VBG going\) \(S \(NP-SBJ \(-NONE- \*-3\)\) \(VP \(TO to\) \(VP \(VB find\) / \(SBAR-PRD \(WHNP-6 \(WP what\)\) \(S \(NP-SBJ-3 \(PRP you\)\) \(VP \(VBP 're\) \(VP \(VBG going\) \(S \(NP-SBJ \(-NONE- \*-3\)\) \(VP \(TO to\) \(VP \(VB find\) \(NP \(-NONE- \*T\*-6\)\) /;
  # line 2556:
  s/\(WHNP \(WP what\)\) \(S \(NP-SBJ \(DT the\) \(-LRB- -LRB-\) \(NNP Federal\) \(NNP Reserve\) \(-RRB- -RRB-\)\) \(VP \(MD will\) \(VP \(VB do\)/\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(DT the\) \(-LRB- -LRB-\) \(NNP Federal\) \(NNP Reserve\) \(-RRB- -RRB-\)\) \(VP \(MD will\) \(VP \(VB do\) \(NP \(-NONE- \*T\*-6\)\)/;
  # line 2965:
  s/\(WHNP \(WP What\)\) \(S \(NP-SBJ-1 \(PRP I\)\) \(VP \(VBP 'm\) \(VP \(VBG trying\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB say\)/\(WHNP-6 \(WP What\)\) \(S \(NP-SBJ-1 \(PRP I\)\) \(VP \(VBP 'm\) \(VP \(VBG trying\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB say\) \(NP \(-NONE- \*T\*-6\)\)/;
  # line 18338:
  s/\(WHNP \(WP what\)\) \(S \(NP-SBJ \(PRP they\)\) \(VP \(VBD had\) \(VP \(VBN hoped\) \(SBAR \(-NONE- 0\) \(S \(-NONE- \*\?\*\)\)\)/\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(PRP they\)\) \(VP \(VBD had\) \(VP \(VBN hoped\) \(SBAR \(-NONE- \*T\*-6\)\)/;
  # line 28261:
  s/\(WHNP \(WP what\)\) \(S \(NP-SBJ \(NNS banks\)\) \(VP \(VBP charge\) \(NP \(DT each\) \(JJ other\)\)/\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ \(NNS banks\)\) \(VP \(VBP charge\) \(NP \(DT each\) \(JJ other\)\) \(NP \(-NONE- \*T\*-6\)\)/;
  # line 28680:
  s/\(WHNP \(WP what\)\) \(S \(NP-SBJ-1 \(NNP Sony\)\) \(VP \(VBD was\) \(S \(NP-SBJ \(-NONE- \*-1\)\) \(VP \(TO to\) \(VP \(VB offer\)/\(WHNP-6 \(WP what\)\) \(S \(NP-SBJ-1 \(NNP Sony\)\) \(VP \(VBD was\) \(S \(NP-SBJ \(-NONE- *-1\)\) \(VP \(TO to\) \(VP \(VB offer\) \(NP \(-NONE- \*T\*-6\)\)/;
  # line 1628:
  s/\(WHNP \(WDT which\) \) \(S \(NP-SBJ-1 \(DT the\) \(JJ Soviet\) \(NN leader\) \)/\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(DT the\) \(JJ Soviet\) \(NN leader\) \)/;
  # line 3642:
  s/\(WHNP \(IN that\) \) \(S \(NP-SBJ-1 \(DT the\) \(NNS Japanese\) \) \(VP \(MD might\) \(VP \(VB want\) \(S \(NP-SBJ \(-NONE- \*-1\) \) \(VP \(TO to\) \(VP \(VB buy\)/\(WHNP-2 \(IN that\) \) \(S \(NP-SBJ-1 \(DT the\) \(NNS Japanese\) \) \(VP \(MD might\) \(VP \(VB want\) \(S \(NP-SBJ \(-NONE- \*-1\) \) \(VP \(TO to\) \(VP \(VB buy\) \(NP \(-NONE- \*T\*-2\) \)/;
  # line 5642:
  s/\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(NNP Bush\) \) \(VP \(VBD allowed\) \(S \(NP-SBJ \(-NONE- \*-1\)/\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(NNP Bush\) \) \(VP \(VBD allowed\) \(S \(NP-SBJ \(-NONE- \*T\*-1\)/;
  # line 7797:
  s/\(NP \(DT The\) \(JJ same\) \(NN evening\) \) \(SBAR \(WHNP-4 \(IN that\) \)/\(NP \(DT The\) \(JJ same\) \(NN evening\) \) \(SBAR \(WHPP-4 \(IN that\) \)/;
  # line 17204:
  s/\(WHNP-3 \(WDT which\) \) \(S \(NP-SBJ \(JJ many\) \(NNS investors\) \) \(VP \(VBP say\) \(SBAR \(-NONE- 0\) \(S \(NP-SBJ-1 \(-NONE- \*-3\)/\(WHNP-3 \(WDT which\) \) \(S \(NP-SBJ \(JJ many\) \(NNS investors\) \) \(VP \(VBP say\) \(SBAR \(-NONE- 0\) \(S \(NP-SBJ-1 \(-NONE- \*T\*-3\)/;
  # line 17563:
  s/\(WHNP \(WP what\) \) \(S \(NP-SBJ \(JJ other\) \(JJ prosecutorial\) \(NNS abuses\) \) \(VP \(MD may\) \(VP \(VB have\) \(VP \(VBN occurred\)/\(WHNP-1 \(WP what\) \(JJ other\) \(JJ prosecutorial\) \(NNS abuses\) \) \(S \(NP-SBJ \(-NONE- \*T\*-1\) \) \(VP \(MD may\) \(VP \(VB have\) \(VP \(VBN occurred\)/;
  # line 25018:
  s/\(NP \(NP \(DT a\) \(NN year\) \) \(SBAR \(WHNP-1 \(IN that\) \)/\(NP \(NP \(DT a\) \(NN year\) \) \(SBAR \(WHPP-1 \(IN that\) \)/;
  # line 27938:
  s/\(WHNP \(WP what\) \) \(S \(NP-SBJ \(DT some\) \(NNS economists\) \) \(VP \(VBP call\) /\(WHNP-1 \(WP what\) \) \(S \(NP-SBJ \(DT some\) \(NNS economists\) \) \(VP \(VBP call\) \(NP \(-NONE- \*T\*-1\) \) /;
  # line 29529:
  s/\(WHNP \(WDT which\) \) \(S \(NP-SBJ \(PRP it\) \) \(VP \(MD will\) \(VP \(VB occupy\) /\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(PRP it\) \) \(VP \(MD will\) \(VP \(VB occupy\) \(NP \(-NONE- \*T\*-1\) \) /;
  # line 34726:
  #s/\(WHADJP \(RB exactly\) \(WP what\)\) \(NN impact\)/\(WHNP \(RB exactly\) \(WDT what\) \(NN impact\)\)/;
  # line 36352:
  s/\(WHNP \(IN that\) \) \(S \(NP-SBJ \(DT the\) \(NN government\) \) \(VP \(VBD feared\) \(SBAR \(WHNP-1 \(-NONE- 0\) \)/\(WHNP-1 \(IN that\) \) \(S \(NP-SBJ \(DT the\) \(NN government\) \) \(VP \(VBD feared\) \(SBAR \(WHNP \(-NONE- 0\) \)/;
  # line 36736:
  s/\(WHNP \(IN that\) \) \(S \(NP-SBJ \(PRP he\) \) \(VP \(VBD estimated\) \(SBAR \(WHNP-2 \(-NONE- 0\) \)/\(WHNP-2 \(IN that\) \) \(S \(NP-SBJ \(PRP he\) \) \(VP \(VBD estimated\) \(SBAR \(WHNP \(-NONE- 0\) \)/;
  # line 37010:
  s/\(WHNP \(IN that\) \) \(S \(NP-SBJ \(DT some\) \(NNS analysts\) \) \(VP \(VBP say\) \(SBAR \(WHNP-1 \(-NONE- 0\) \) /\(WHNP-1 \(IN that\) \) \(S \(NP-SBJ \(DT some\) \(NNS analysts\) \) \(VP \(VBP say\) \(SBAR \(WHNP \(-NONE- 0\) \) /;
   # line 37777:
  s/\(SBAR \(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ-2 \(-NONE- \*-1\) \)/\(SBAR \(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ-2 \(-NONE- \*T\*-1\) \)/;
  # line 39576:
  s/\(WHNP \(WDT which\) \) \(S \(NP-SBJ \(PRP they\) \) \(VP \(VBP sell\) /\(WHNP-1 \(WDT which\) \) \(S \(NP-SBJ \(PRP they\) \) \(VP \(VBP sell\) \(NP \(-NONE- \*T\*-1\) \)/;

  ## false VBN -> should be VBD...
  # line 41:
  s/\(, ,\) \(VP \(VBN reached\) \(NP \(-NONE- \*T\*-/\(, ,\) \(VP \(VBD reached\) \(NP \(-NONE- \*T\*-/;
  # line 4257:
  s/2\) \) \(VP \(VBN forced/2\) \) \(VP \(VBN forced/;
  # line 6124:
  s/\(VP \(VBN suspended\) \(NP \(NN stock-index\)/\(VP \(VBD suspended\) \(NP \(NN stock-index\)/;
  # line 6815:
  s/\(VP \(VBN involved\) \(NP \(JJ critical\)/\(VP \(VBD involved\) \(NP \(JJ critical\)/;
  # line 7676:
  s/\(NN week\) \) \(VP \(VBN succeeded\)/\(NN week\) \) \(VP \(VBD succeeded\)/;
  # line 10076:
  s/\(NNS completions\) \) \(VP \(VBN lagged\)/\(NNS completions\) \) \(VP \(VBD lagged\)/;
  # line 10572:
  s/\(RB ago\) \) \(VBN acquired\)/\(RB ago\) \) \(VBD acquired\)/;
  # line 11963:
  s/\(, ,\) \) \(VP \(VBN announced\)/\(, ,\) \) \(VP \(VBD announced\)/;
  # line 12946:
  s/3\) \) \(VP \(VBN generated/3\) \) \(VP \(VBD generated/;
  # line 13609:
  s/NNS researchers\) \) \(VP \(VBN agreed/NNS researchers\) \) \(VP \(VBD agreed/;
  # line 15066:
  s/1\) \) \(VP \(VBN started/1\) \) \(VP \(VBD started/;
  # line 15591, 28900:
  s/1\) \) \(VP \(VBN changed/1\) \) \(VP \(VBD changed/;
  # line 15961:
  s/1\) \) \(VP \(VBN opened/1\) \) \(VP \(VBD opened/;
  # line 18772:
  s/NNP KKR\) \) \(VP \(VBN restructured/NNP KKR\) \) \(VP \(VBD restructured/;
  # line 19265:
  s/1\) \) \(VP \(VBN included/1\) \) \(VP \(VBD included/;
  # line 19680:
  s/VBN become\) \(NP-PRD \(JJ overnight/VBD become\) \(NP-PRD \(JJ overnight/;
  # line 19748:
  s/NN bank\) \) \(VP \(VBN disclosed\)/NN bank\) \) \(VP \(VBD disclosed\)/;
  # line 21201:
  s/NP Goodson\) \) \(VP \(VBN bought/NP Goodson\) \) \(VP \(VBD bought/;
  # line 23901:
  s/2\) \) \(VP \(VBN opened/2\) \) \(VP \(VBD opened/;
  # line 24689:
  s/1\) \) \(VP \(VBN traded/1\) \) \(VP \(VBD traded/;
  # line 24700:
  s/3\) \) \(VP \(VBN stretched/3\) \) \(VP \(VBD stretched/;
  # line 26286:
  s/VBN set\) \(NP \(NNS plans/VBD set\) \(NP \(NNS plans/;
  # line 28510:
  s/1\) \) \(VP \(VBN made/1\) \) \(VP \(VBD made/;
  # line 32752:
  s/NNS eggs\) \) \(VP \(VBN come/NNS eggs\) \) \(VP \(VBD come/;
  # line 33883:
  s/NNP II\) \) \(VP \(VBN ended/NNP II\) \) \(VP \(VBD ended/;
  # line 38619:
  s/1\) \) \(VP \(VBN set\) \(PRT \(RP off/1\) \) \(VP \(VBD set\) \(PRT \(RP off/;
  # line 38773:
  s/NN company\) \) \(VP \(VBN valued/NN company\) \) \(VP \(VBD valued/;
  # line 38814:
  s/1\) \) \(VP \(VBN induced/1\) \) \(VP \(VBD induced/;
  # line 39070:
  s/NN market\) \) \(VP \(VBN stabilized/NN market\) \) \(VP \(VBD stabilized/;
  # line 39264:
  s/PRP it\) \) \(VP \(VBN opened/PRP it\) \) \(VP \(VBD opened/;
  # line 39579:
  s/NN analyst\) \) \(VP \(VBN estimated/NN analyst\) \) \(VP \(VBD estimated/;
  # line 39619:
  s/RB finally\) \) \(VP \(VBN opened/RB finally\) \) \(VP \(VBD opened/;
  # line 39623:
  s/NN market\) \) \(VP \(VBN opened/NN market\) \) \(VP \(VBD opened/;
  # line 39779:
  s/RB never\) \) \(VBN reopened/RB never\) \) \(VBD reopened/;

  # line ...: weird 'have issues forecasts' error -> should be 'issued' (would get caught by annotCats)
  s/\(VP *\(NNS issues/\(VP \(VBN issued/;

  # false NN -> should be VB/VBZ
  s/\(VP  *\(NNS  *([^\)]*)\)/\(VP \(VBZ \1\)/g;
  s/\(VP  *\(NN  *([^\)]*)\)/\(VP \(VB \1\)/g;

  # false VBD -> should be VBN...
  s/\(VBD ((?!received)[^\)]*)\) *\(NP *\(-NONE- \*\) \)/\(VBN \1\) \(NP \(-NONE- \*\) \)/g;

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug($step++, "   $_");

    # line 3828, 19200 (incl 01): false (NP ... PRP POS) ... -> should be (NP ...) (VP VBZ ...)
    s/{(.*) <(NP[^>]*\[PRP[^\]]*\]) *\[POS *'s\] *> *(<VP.*)}/\(\1 <\2> \{VP <VBZ 's> \3\}\)/;
#    # line 2247, ...: false POS 's following letter -> should be NNS (also wrong, but avoids later problems)
#    s/{((?!NP).*<NP[^>]*\[(?:NN|PRP)[^\]]*\] *)\[POS *'s\]( *>.*)}/{\1 \[NNS 's\]\2}/;
#    #s/{(NP.*<(?:NN|PRP)[^ ]* [A-Z]>) *<POS *'s> *}/{\1 <NNS 's>}/;

#    # ???? does not seem to exist!
#    # line 24513:
    s/{(SBAR) *<(WHNP-1 [^>]*)> *<(S) ([^>]*)> <(VP [^>]*)> *}/{\1 <\2> <\3 \4 [\5]>}/;

    # line 6603, 18358 (incl 01): false NP at beginning of VP -> should be VP
    s/\((VP *)\{NP( <VBZ (?:kills|blames)>.*)\}/\(\1\{VP\2}/;

    ## conj scoping errors...
    # line 3911 (incl 01):
    s/(<VB have>[^\(]*){VP *(<VP *\[VBN made\].*) *(<CC.*<VP *\[VB discard\].*)}/{VP \1\2\}\3/;
    # line 5940 (incl 01):
    s/(<VB resume>[^\(]*){NP *(<NP *\[NN growth\].*) *(<CC.*)<NP *(\[VB service\].*)}/{VP \1\2\} \3<VP \4/;
    # line 10683 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN purchased\].*) *(<CC.*<VP *\[VBZ plans\].*)}/{VP \1\2\}\3/;
    # line 10860 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN started\].*) *(<CC.*<VP *\[VBZ has\].*)}/{VP \1\2\}\3/;
    # line 18311 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN met\].*) *(<CC.*<VP *\[VBZ is\].*)}/{VP \1\2\}\3/;
    # line 29504 (incl 01):
    s/(<VBP have>[^\(]*){VP *(<VP *\[VBN plunged\].*) *(<CC.*<VP *\[PP[^>]*\[VBP are\].*)}/{VP \1\2\}\3/;
    # line 29706 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN reacted\].*) *(<CC.*<VP *\[ADV[^>]*\[VBZ is\].*)}/{VP \1\2\}\3/;
    # line 29951 (incl 01):
    s/(<VBP have>[^\(]*){VP *(<VP *\[VBN outgrown\].*) *(<CC.*<VP *\[VBP have\].*)}/{VP \1\2\}\3/;
    # line 30304 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN lost\].*) *(<CC.*<VP *\[VBZ has\].*)}/{VP \1\2\}\3/;
    # line 31705 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(<VP *\[VBN licensed\].*) *(<CC.*<VP *\[VBZ plans\].*)}/{VP \1\2\}\3/;
    # line 41333 (incl 01):
    s/(<VBP have>[^\(]*){VP *(<VP *\[VBN read\].*) *(<CC.*<VP *\[VBP estimate\].*)}/{VP \1\2\}\3/;
    # line 41381 (incl 01):
    s/(<VBZ has>[^\(]*){VP *(.*<VP *\[VB hurt\].*) *(<CC.*<VP *\[VBZ has\].*)}/\(VP {VP \1\2\}\3\)/;
    # line 40563 (incl 01):
    s/{VP *(<VBD had> *<VP *\[VBN silted\][^>]*>)( *<CC.*<VP *\[ADV[^>]*\[VBD was\].*)}/\(VP {VP \1\}\2\)/;

    # line 10807 (incl 01):
    s/{(NP[^ ]*) *<(NP[^>]*)> *<VBZ *'s> (.*)}/{\1 <\2 \[POS 's\]> \3}/;

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

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug($step++, "   $_");

    # false VBZ 's anywhere in NP -> should be POS
    while ( s/{(NP.*)<VBZ *'s>((?! *<DT).*)}/{\1<POS 's>\2}/ ){}
    # false POS 's not at end of NP sections off beginning
    s/{(NP[^ ]*) (.*<.*<POS *'s>)(.*<.*)}/\(\1 \{NP \2\}\3\)/;
    # false VBZ non-'s at end of NP -> should be NNS
    s/{(NP.*)<VBZ ([^>]*)> *}/{\1<NNS \2>}/;
    # false VB anywhere immediately beneath NP -> should be NN
    while ( s/{(NP.*)<VB ([^>]*)>(.*)}/{\1<NN \2>\3}/ ){}
    # false VBG conjoined in NP -> should be NN
    s/{(NP.*)<VBG ([^>]*)> *<CC ([^>]*)> *<VBG ([^>]*)> *}/{\1<NN \2> <CC \3> <NN \4>}/;
    # false VBG at end of NP -> should be NN
    s/{(NP.*)<VBG ([^>]*)> *}/{\1<NN \2>}/;

    # false S -> should be S-ADV
    s/{(S *<)S( [^>]*> *<,[^>]*> .*<NP.*)}/{\1S-ADV\2}/;

    # false VBD -> should be VBN...
    s/({NP[^\}]*<VP[^\]]*)VBD/\1VBN/;

    # false VBN post-modifying NP without -NONE- -> should be with -NONE-...
    if (! $PSG) {
      s/(\(NP[^\(\)]*\{VP *<VBN[^>]*> *(?!( *<PRT[^>]*>|( *<[^>]*>)*| *<PP[^ ]* *\[(RB|RP|IN)[^\]]*\])?[^>\]]*\[-NONE- \*))/\1 <NP \[-NONE- \*\]>/;
	}
	
    # false VBN in root S -> should be VBD (dependent on S-ADV above)...
    s/(^[^\)>\]]*\{S .*<NP[^>]*>(?! *<VP[^>\]]*MD).*<VP[^>\]]*)VBN(?! married|[^\]]*\] *\[NP *\[-NONE- *\*)(.*\})/\1VBD\2/;

	if ($PSG) {
		debug($step,"?? $_");
		# false RB yet -> should be CC...
		s/{([^ ]*) <([^- ]*(?![A-Z]))([^>]*)>((?: <[^ ]* (?:``|,|'')>)*) <[^\]>]* yet[\]]*> <\2([^>]*)>}/{\1 <\2\3>\4 <CC yet> <\2\5>}/;

		# false SBAR WH without a trace gets trace added...
		s/{(SBAR[^ ]*) <(WHADVP(?:(?!-[0-9]+)[^ ])*) ([^>]*)> <(S[^ ]*) ([^>]*)>}/{\1 <\2-0000 \3> <\4 \5 \[ADVP \[-NONE- \*T\*-0000\]\]>}/;

		# false extraposed SBAR too deep in constit gets pushed up...
		s/(\[-NONE- \*ICH\*(-[0-9]+)\].*){(.*) <(SBAR[^ ]*\2.*)>}/\1\{\3\} <\4>/;

		# never happens
		# s/{(SBAR[^ ]*) <(WHADVP[^ ]*)(-[0-9]+(?![0-9]))([^>]*)> <(S[^ ]*)(?![^>]*\[-NONE- \*T\*\3\][^>]*)([^>]*)>}/{\1-mody <\2\3\4> <\1 \[-NONE- \*T\*\3\]\2>}/;
	}
	
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

  # false VBD -> should be VBN (dependent on VBZ->POS change above)...
  s/(( be| being| been| is| was|VBZ 's| are| were| 're)\)[^\)]*)VBD/\1VBN/g;
  # false VBD -> should be VBN (dependent on VBZ->POS change above)...
  s/(( have| 've| having| has|VBZ 's| had| 'd)\)[^\)]*)VBD/\1VBN/g;

  ## for each constituent...
  while ( $_ =~ /\([^\(\)]*\)/ ) {
    ## convert outer parens to braces...
    s/\(([^\(\)]*)\)/{\1}/;
    #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
    debug($step++, "   $_");

    # false VBD in conj following initial VBN -> should be VBN, for every subsequent conjunct
    while ( s/{(VP[^ ]* *<VP[^>\]]*VBN[^>]*>(?= .*<CC[^>]*>.*\}).*<VP[^>\]]*)VBD(.*)}/{\1VBN\2}/ ) { }

    # false VBN in conj following initial VBD -> should be VBD, for every subsequent conjunct
    while ( s/{(VP[^ ]* *<VP[^>\]]*VBD[^>]*>(?= .*<CC[^>]*>.*\}).*<VP[^>\]]*)VBN(.*)}/{\1VBD\2}/ ) { }

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


  ## output...
  print $_;
}
