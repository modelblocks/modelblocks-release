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

################################################################################
#
#  i. Macros & variables
#
################################################################################

.SECONDARY:
.SUFFIXES:
.SECONDEXPANSION:

THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))

SCRIPTS := $(THISDIR)scripts
SRC     := $(THISDIR)src
#GENMODEL := genmodel
#BIN := bin
#GENMODEL := $(CURDIR)/genmodel
#BIN := $(CURDIR)/bin
#GENMODEL := $(THISDIR)genmodel
#BIN := $(THISDIR)../wsjparse/bin

all:  wsj23-comparativized-gcg15-nofail.wsj02to21-comparativized-gcg15-1671-5sm.fullberk.parsed.syneval

################################################################################
#
#  1. GCG reannotation
#
#  to construct the following file types:
#    <x>.<y>.linetrees    : treebank-tagset phase structure trees, one sentence per line, bracketed by parens
#
################################################################################

#### misc reannotation (there are multiple copies in case we want to do more than one w/o cycles)
%.fixed.linetrees:  $(SCRIPTS)/make-trees-fixed.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
%.nommed.linetrees:  $(SCRIPTS)/make-trees-nommed.pl  %.linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
.PRECIOUS: %.linetrees
%.linetrees:  $(SCRIPTS)/make-trees-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).linetrees
	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@

#### gcg reannotation
.PRECIOUS: %.linetrees  ## NOTE: inline seds dispose of reannotation failures, then remove -f tags, then shift -l tags to end
%.linetrees:  $(SCRIPTS)/annotate-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).fixed.nommed.linetrees  $(SCRIPTS)/killUnaries.pl
	cat $(word 2,$^)  |  perl $(word 1,$^)  |  perl $(word 3,$^) -l  |  perl -pe 's/.*\([-A-Z]+ .*//'  |  perl -pe 's/\(([^ ]*)-f[^ ]*/\(\1/g' | perl -pe 's/\(([^ ]*)-l([A-Z])([^ ]*)/\(\1\3-l\2/g;'  >  $@

#### remove entries with no reannotation
%.nofail.linetrees:  %.linetrees
	cat $<  |  grep -v '^$$'  >  $@

#### remove gcg reannotation
%.nodashtags.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/([^\(])\-[^ \)]+ /\1 /g'  >  $@

#### annotate branch and depth
%.bd.linetrees:  %.linetrees  scripts/annotateDepth.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### remove unary branches
%.nounary.linetrees:  %.linetrees  scripts/killUnaries.pl
	cat $<  |  perl $(word 2,$^)  >  $@

################################################################################
#
#  2. Propositional content formatting items
#
#  to construct the following file types:
#    <x>.dsgraphs : deep syntactic dependency graphs, one sentence per line, with eventuality-specific dependencies delimited by spaces
#    <x>.tgraphs  : same, but restricted to tree structure for eval purposes
#    <x>.<y>grapheval : evaluation report for various dependency types
#
################################################################################

#### obtain sentence-aligned space-delimited variable (semantic) dependencies
.PRECIOUS: %.protographs
%.protographs:  %.linetrees  $(SCRIPTS)/trees2protographs.py
	cat $<  |  python3 $(word 2,$^)  |  perl -pe 's/0,:/0,Pk/g'  >  $@

#### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.cuegraphs
%.cuegraphs:  %.protographs  $(SCRIPTS)/protographs2cuegraphs.pl
	cat $<  |  perl -p $(word 2,$^)  >  $@

#### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.varconts
%.varconts:  %.cuegraphs  $(SCRIPTS)/cuegraphs2varconts.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### obtain sentence-aligned space-delimited deep syntactic (tree-like) dependencies
# %.dsgraphs:  %.protographs
# 	cat $<  |  perl -pe 's/([0-9]+)[crs]/\1/g'  >  $@
%.dsgraphs:  %.protographs  $(SCRIPTS)/protographs2dsgraphs.pl
	cat $<  |  perl -p $(word 2,$^)  >  $@

# #### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
# .PRECIOUS: %.vgraphs
# %.vgraphs:  %.protographs  $(SCRIPTS)/protographs2vgraphs.pl
# 	cat $<  |  perl -p $(word 2,$^)  >  $@

#### obtain tree-restricted sentence-aligned space-delimited eventuality (semantic) dependencies
.PRECIOUS: %.tgraphs
%.tgraphs:  %.linetrees  $(SCRIPTS)/trees2dsgraphs.py
	cat $<  |  python3 $(word 2,$^) -t  |  perl -pe 's/([0-9]+)[crs]/\1/g'  >  $@

#### obtain eval by running evaluator on gold and hypoth elementary predication dependencies
#%grapheval:  $(SCRIPTS)/depeval.py  $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(basename $$(basename $$(basename $$(basename $$(notdir $$*)))))))$$(suffix $$*)graphs  %graphs
#	python3 $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@
%grapheval:  $(SCRIPTS)/depeval.py \
             $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(basename $$(basename $$(basename $$(basename $$(notdir $$*)))))))$$(suffix $$*)graphs \
             $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$(basename $$*))))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*)graphs
	python3 $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@

