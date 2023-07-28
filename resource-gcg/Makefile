################################################################################
##                                                                            ##
##  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                            ##
##  ModelBlocks is free software: you can redistribute it and/or modify       ##
##  it under the terms of the GNU General Public License as published by      ##
##  the Free Software Foundation, either version 3 of the License, or         ##
##  (at your option) any later version.                                       ##
##                                                                            ##
##  ModelBlocks is distributed in the hope that it will be useful,            ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##  GNU General Public License for more details.                              ##
##                                                                            ##
##  You should have received a copy of the GNU General Public License         ##
##  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      ##
##                                                                            ##
################################################################################


# TO MAKE simplewikiC1:  cat wikisemc1.casp.toktrees | perl ../resource-linetrees/scripts/editabletrees2linetrees.pl| python3 scripts/senttrees2discexprs.py -s41 -s82 -s91 -s20 -s35 -s49 -s114  -s61 -s40 -s127  > eraseme1
# TO MAKE simplewikiC2:  cat wikisemc2.casp.toktrees | perl ../resource-linetrees/scripts/editabletrees2linetrees.pl| python3 scripts/senttrees2discexprs.py -s61 -s125 -s43 -s50 -s72 -s83 -s85 -s113 -s127 > eraseme2


################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

all:  wsj23-comparativized-gcg15-nofail.wsj02to21-comparativized-gcg15-1671-5sm.fullberk.parsed.syneval

ifndef MAKED
$(info Building source code dependencies using recursive sub-make to $(RESOURCE-GCG)/Makefile...)
## create dependencies for files in python, src, include directories
DUMMY := $(shell $(MAKE) MAKED=1 MAKECONFIG=1 $(subst .py,.d,$(wildcard $(GCG-SCRIPTS)/*.py)) $(subst .cpp,.d,$(wildcard $(SRC)/*.cpp)))
$(info Source code dependencies built and loaded.)
endif

## include those dependencies
include $(wildcard $(RESOURCE-GCG)/*/*.d)

ifdef MAKED
#### python dependencies
.PRECIOUS:  $(GCG-SCRIPTS)/%.d
$(GCG-SCRIPTS)/%.d:  $(GCG-SCRIPTS)/%.py
	echo '$@: \'  >  $@   #' ##
	egrep '^from|^import' $<  |  sed 's/^from \(.*\) import .*/$$(wildcard $$(GCG-SCRIPTS)\/\1.py) \\/;s/^import \(.*\)$$/$$(wildcard $$(GCG-SCRIPTS)\/\1.py) \\/'  >>  $@
	echo ''  >>  $@
	cat $@  |  sed 's/\.d:/\.py:/'  >>  $@
endif

################################################################################
#
#  GCG reannotation
#
#  to construct the following file types:
#    <x>.<y>.linetrees    : treebank-tagset phase structure trees, one sentence per line, bracketed by parens
#
################################################################################

#### misc reannotation (there are multiple copies in case we want to do more than one w/o cycles)
%.comparativized.linetrees:  $(GCG-SCRIPTS)/make-trees-comparativized.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
%.fixed.senttrees:  $(GCG-SCRIPTS)/make-trees-fixed.pl  %.senttrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
%.nommed.linetrees:  $(GCG-SCRIPTS)/make-trees-nommed.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
%.linetrees.validation:  %.linetrees  $(GCG-SCRIPTS)/validate-linetrees.py  $(GCG-SCRIPTS)/validate-linetrees.d
	cat $<  |  python $(word 2,$^)  >  $@

### THIS HAPPENS ALL THE TIME; JUST CREATE MAKE ITEM IF YOU NEED IT
# .PRECIOUS: %.linetrees
#%.linetrees:  $(GCG-SCRIPTS)/make-trees-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).linetrees
#	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
  

# obsolete, use nommed if desired
# %.decoupled.linetrees:  $(GCG-SCRIPTS)/annotate-$$(subst .,,$$(suffix $$*)).pl $$(basename %).fixed.linetrees $(LTREES-SCRIPTS)/killUnaries.pl
# 	cat $(word 2,$^)  |  perl $(word 1,$^)  |  perl $(word 3,$^) -l  |  perl -pe 's/.*\([-A-Z]+ .*//'  |  perl -pe 's/\(([^ ]*)-f[^ ]*/\(\1/g' | perl -pe 's/\(([^ ]*)-l([A-Z])([^ ]*)/\(\1\3-l\2/g;'  >  $@
        
#### NOTE: inline seds dispose of reannotation failures, then remove -f tags, then shift -l tags to end
.PRECIOUS: %.senttrees
%.senttrees:  $(GCG-SCRIPTS)/annotate-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).fixed.senttrees  $(LTREES-SCRIPTS)/killUnaries.pl $(GCG-SCRIPTS)/makefailtrees.py
	cat $(word 2,$^)  |  perl $(word 1,$^)  |  perl $(word 3,$^) -l  |  python3 $(word 4,$^) |  perl -pe 's/\(([^ ]*)-f[^ ]*/\(\1/g' |  perl -pe 's/\(([^ ]*)-l([A-Z])([^ ]*)/\(\1\3-l\2/g;'  >  $@
#cat $(word 2,$^)  |  perl $(word 1,$^)  |  perl $(word 3,$^) -l  |  perl -pe 's/.*\([^a-z]+ .*//'  |  perl -pe 's/\(([^ ]*)-f[^ ]*/\(\1/g' |  perl -pe 's/\(([^ ]*)-l([A-Z])([^ ]*)/\(\1\3-l\2/g;'  >  $@
## add !ARTICLE delimiters...
%artdelim.linetrees:  $(GCG-SCRIPTS)/addDelimiter.py  $$(basename %).linetrees
	cat $(word 2,$^)  |  python $(word 1,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
## add morph tags...
%morphed.senttrees:  $(GCG-SCRIPTS)/make-trees-morphed.pl  $$(basename %).senttrees
	cat $(word 2,$^)  |  perl $(word 1,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
## remove morph tags for eval...
%.nox.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/[-+]x[^-} ][^ |]*[|][^- ]*//g'  >  $@

## dummy linker for casp standard when no coref is available...
%.linked.senttrees:  %.senttrees  genmodel/$$(word 1,$$(subst ., ,$$(notdir $$*))).nocoref
	cat $<  >  $@

## casp currently set to gcg15 + long + linked + morphed (without -x)...
.PRECIOUS: %casp.senttrees
%casp.senttrees:  $(GCG-SCRIPTS)/make-trees-morphed.pl  $$(basename %).gcg15.long.linked.senttrees
	cat $(word 2,$^)  |  perl $(word 1,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
#	cat $(word 2,$^)  |  perl $(word 1,$^) -x $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
.PRECIOUS: %caspu.senttrees
%caspu.senttrees:  $(GCG-SCRIPTS)/make-trees-morphed.pl  $$(basename %).gcg16.long.linked.senttrees
	cat $(word 2,$^)  |  perl $(word 1,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
#	cat $(word 2,$^)  |  perl $(word 1,$^) -x $(subst _, ,$(subst .,,$(suffix $*)))  >  $@

#### remove entries with no reannotation
%.noempty.linetrees:  %.linetrees
	cat $<  |  grep -v '^$$'  >  $@

%.nofail.linetrees: %.linetrees
	cat $< | grep -v '(FAIL' > $@
%.goldfailerr.senttrees: %.senttrees
	cat $< | perl -pe 's/^\(FAIL.*/(FAIL err)/' > $@

%.noqp.linetrees:  %.linetrees  $(GCG-SCRIPTS)/remove-qps.pl
	cat $<  |  perl $(word 2,$^)  >  $@
%.fixdollar.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/\(N \$$\)/\(N-aD \$$\)/g'  >  $@

#### annotate branch and depth
%.bd.linetrees:  %.linetrees  $(GCG-SCRIPTS)/annotateDepth.py
	cat $<  |  python3 $(word 2,$^)  >  $@

define ParamVal
FG:
  kernel: ModelOpt
  text: Filler-gap transform (inserts stack elements for long-distance dependencies)
  value: fg
endef
#### apply filler-gap transform
%.fg.linetrees:  %.linetrees  $(GCG-SCRIPTS)/annotateFGTrans.pl
	cat $<  |  perl $(word 2,$^)  >  $@

#### apply tree-based filler-gap transform
%.fg.t.linetrees:  %.linetrees  $(GCG-SCRIPTS)/annotateFGTrans.pl
	cat $<  |  perl $(word 2,$^) -t >  $@

#### apply semantic left-corner transform
%.semlctrans.linetrees: %.linetrees ../incrsem/scripts/semlctrans.pl
	cat $< | perl $(word 2,$^) > $@

#### lengthen trees to full gcg derivation...
%.long.senttrees:  %.senttrees  $(GCG-SCRIPTS)/lengthentrees.py  $(GCG-SCRIPTS)/lengthentrees.d
	cat $<  |  python3 $(word 2,$^)  >  $@

#### Remove traces
%.nt.linetrees: %.linetrees
	cat $^ | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\t/ /g;s/\([^ ]\)(/\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' > $@

#### Remove subcategorization information (only from preterminals)
%.nounary.nosubcat.linetrees: %.nounary.linetrees
	cat $^ | perl -pe 's/\(([^- ]+)-[^ ]+ ([^ \)]*)\)/\(\1 \2\)/g' > $@


################################################################################
#
#  Propositional content formatting items
#
#  to construct the following file types:
#    <x>.dsgraphs : deep syntactic dependency graphs, one sentence per line, with eventuality-specific dependencies delimited by spaces
#    <x>.tgraphs  : same, but restricted to tree structure for eval purposes
#    <x>.<y>grapheval : evaluation report for various dependency types
#
################################################################################

#### annotation release format...
%.toktrees:  $(GCG-SCRIPTS)/port-semantics.perl  %.annot-syn  %.annot-sem
	perl $^  >  $@
%.senttrees:  %.toktrees  $(LTREES-SCRIPTS)/editabletrees2linetrees.pl
	cat $<  |  perl $(word 2,$^)  >  $@
#### indentation formats for annotation...
%.fromtoktrees.annot-syn:  %.toktrees
	cat $<  |  perl -pe 's/-[mnstuw]+[0-9]+r?//g'  >  $@
%.fromtoktrees.annot-sem:  %.toktrees  $(GCG-SCRIPTS)/dedent.perl  $(GCG-SCRIPTS)/bare-ends.perl  $(GCG-SCRIPTS)/yoink-word-rightward.perl  $(GCG-SCRIPTS)/nathans-numberer.perl
	cat $(word 1,$^)  |  perl $(word 2,$^) | perl $(word 3,$^) | perl $(word 4,$^) | perl $(word 5,$^) >  $@

#### auto lengthener repair...
%.lengthrepaired.senttrees:  %.annot-syn  $(LTREES-SCRIPTS)/editabletrees2senttrees.pl  $(GCG-SCRIPTS)/fix_lengthener_errors.py
	cat $<  |  perl $(word 2,$^)  |  python3 $(word 3,$^)  >  $@

#### DEPRECATED: obtain sentence-aligned space-delimited variable (semantic) dependencies
.PRECIOUS: %.protographs
%.protographs:  %.linetrees  $(GCG-SCRIPTS)/trees2protographs.py
	cat $<  |  python3 $(word 2,$^)  |  perl -pe 's/0,:/0,Pk/g'  >  $@

#### DEPRECATED: obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.cuegraphs
%.cuegraphs:  %.linetrees  $(GCG-SCRIPTS)/linetrees2cuegraphs.d  $(GCG-SCRIPTS)/linetrees2cuegraphs.py
	cat $<  |  sed 's/\^g//g' | python $(word 3,$^)  >  $@
#%.cuegraphs:  %.protographs  $(GCG-SCRIPTS)/protographs2cuegraphs.pl
#	cat $<  |  perl -p $(word 2,$^)  >  $@

#### obtain discourse-aligned space-delimited cued-association graphs
.PRECIOUS: %.discgraphs
%.discgraphs:  %.senttrees  $(GCG-SCRIPTS)/senttrees2discgraphs.d  $(GCG-SCRIPTS)/senttrees2discgraphs.py
	cat $<  |  sed 's/\^g//g' | python2 $(word 3,$^) -e  >  $@
# not for end-users, just for debugging:
%.verbose-discgraphs:  %.senttrees  $(GCG-SCRIPTS)/senttrees2discgraphs.d  $(GCG-SCRIPTS)/senttrees2discgraphs.py
	cat $<  |  sed 's/\^g//g' | python2 $(word 3,$^) -d -e  >  $@

# #### obtain pdf drawing of discgraph   (neato/dot)
# %.discgraphs.pdf:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discgraphspdf.py
# 	cat $<  |  python $(word 2,$^)  |  dot -Tpdf  >  $@

#### obtain tex drawing of discgraph
%.discgraphs.tex:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discgraphstex.d  $(GCG-SCRIPTS)/discgraphs2discgraphstex.py
	cat $<  |  python2 $(word 3,$^)  >  $@

#### obtain pdf from tex
%.pdf:  %.tex
	cd $(dir $<);  pdflatex $(notdir $<)

#### obtain discourse-aligned lambda expressions
%.discgraphs.formerrors:  %.discgraphs  $(GCG-SCRIPTS)/discgraphvalidator.d  $(GCG-SCRIPTS)/discgraphvalidator.py
	cat $<  |  python2 $(word 3,$^)  >  $@

#### obtain discourse-aligned lambda expressions
%.discexprs:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discexprs.d  $(GCG-SCRIPTS)/discgraphs2discexprs.py  bin/indent
	cat $<  |  python2 $(word 3,$^)  |  $(word 4,$^)  >  $@
#### obtain discourse-aligned lambda expressions, debugging stuff
%.verbose-discexprs:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discexprs.d  $(GCG-SCRIPTS)/discgraphs2discexprs.py  bin/indent
	cat $<  |  python2 $(word 3,$^) -d  |  $(word 4,$^)  >  $@
%.verbose-discexprs.last.discgraphs:  %.verbose-discexprs
	cat $<  |  grep 'GRAPH:'  |  tail -n1  |  perl -pe 's/ *GRAPH: *//'  >  $@
# also for debugging
%.autoinduced.discgraphs:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discexprs.d  $(GCG-SCRIPTS)/discgraphs2discexprs.py
	cat $<  |  python $(word 3,$^) -g  |  grep 'GRAPH:'  |  perl -pe 's/ *GRAPH: *//'  >  $@

#### obtain discgraphs after scope induction & assorted normalization, ready to translate to discexprs
%.autonormed.discgraphs:  %.discgraphs  $(GCG-SCRIPTS)/discgraphs2discexprs.d  $(GCG-SCRIPTS)/discgraphs2discexprs.py
	cat $<  |  python $(word 3,$^) -n  |  grep 'GRAPH:'  |  perl -pe 's/ *GRAPH: *//'  >  $@

#### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.refconts
%.refconts:  %.cuegraphs  $(GCG-SCRIPTS)/cuegraphs2refconts.py
	cat $<  |  python3 $(word 2,$^) | sort -g >  $@

#### obtain sentence-aligned space-delimited deep syntactic (tree-like) dependencies
# %.dsgraphs:  %.protographs
# 	cat $<  |  perl -pe 's/([0-9]+)[crs]/\1/g'  >  $@
%.dsgraphs:  %.protographs  $(GCG-SCRIPTS)/protographs2dsgraphs.pl
	cat $<  |  perl -p $(word 2,$^)  >  $@

# #### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
# .PRECIOUS: %.vgraphs
# %.vgraphs:  %.protographs  $(GCG-SCRIPTS)/protographs2vgraphs.pl
# 	cat $<  |  perl -p $(word 2,$^)  >  $@

#### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.tgraphs
%.tgraphs:  %.linetrees  $(GCG-SCRIPTS)/trees2dsgraphs.py
	cat $<  |  python3 $(word 2,$^) -t  |  perl -pe 's/([0-9]+)[crs]/\1/g'  >  $@

#### obtain eval by running evaluator on gold and hypoth elementary predication dependencies
#%grapheval:  $(GCG-SCRIPTS)/depeval.py  $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(basename $$(basename $$(basename $$(basename $$(notdir $$*)))))))$$(suffix $$*)graphs  %graphs
#	python3 $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@
%grapheval:  $(GCG-SCRIPTS)/depeval.py \
             $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(basename $$(basename $$(basename $$(basename $$(notdir $$*)))))))$$(suffix $$*)graphs \
             $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$(basename $$*))))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*)graphs
	python3 $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@

