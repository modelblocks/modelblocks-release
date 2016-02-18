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
#  INCLUDES
#
#  Makefiles that include this one should also include the following files:
#
#  ../general/Makefile
#
################################################################################

################################################################################
#
#  i. Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

#The directory variables break when this makefile is called by another.
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SRCMODEL := $(THISDIR)srcmodel
SCRIPTS  := $(THISDIR)scripts



################################################################################
#
#  ii. User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

viewconfig:
	@echo ''
	@echo 'Here is a list of the required user config files for this project and their current values:'
	@echo ''
	@echo '../config/user-dundee-directory: '$(DUNDEE)
	@echo ''

MSG1 := The current config file 
MSG2 :=  points to an incorrect location (
MSG3 := ). Fix it before re-running make

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which may contain an incorrect default value.
Fix it before re-running make.

endef

ifndef MAKECONFIG
CONFIG := ../config/user-dundee-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
DUNDEE := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(DUNDEE))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(DUNDEE)$(MSG3))
endif
endif

%/user-dundee-directory.txt: | %
	echo '/home/corpora/original/english/dundee' > $@


    
################################################################################
#
#  iii. Directory recipes
#
################################################################################
    
.PRECIOUS: golddata
golddata:
	mkdir golddata

    
    
    
################################################################################
#
#  1. Gold-standard annotated dundee corpus
#
################################################################################

.PRECIOUS: genmodel/dundee%.sents
genmodel/dundee%.sents: $(SCRIPTS)/builddundeecorpus.py $(SCRIPTS)/convert_latin-1.py | genmodel
	cat $(DUNDEE)/tx$*wrdp.dat | python $(word 1,$^) > $@.raw
	cat $@.raw | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 2,$^) > $@
	rm -f $@.raw

.PRECIOUS: genmodel/dundee%.notoksents
#non-tokenized version of dundee (for ngram calculation)
genmodel/dundee%.notoksents: $(DUNDEE)/tx%wrdp.dat $(SCRIPTS)/buildnotokdundeecorpus.py $(SCRIPTS)/convert_latin-1.py | genmodel
	cut -d" " -f1 < $(word 1,$^) | python $(word 2,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 3,$^) > $@

.PRECIOUS: genmodel/dundee.sents
genmodel/dundee.sents: $(foreach sect,$(DUNDEESECTS),genmodel/dundee$(sect).sents) | genmodel
	cat $^ > $@

.PRECIOUS: genmodel/dundee.notoksents
genmodel/dundee.notoksents: $(foreach sect,$(DUNDEESECTS),genmodel/dundee$(sect).notoksents) | genmodel
	cat $^ > $@

.PRECIOUS: %dundee.stripped.linetrees
%dundee.stripped.linetrees: $(SRCMODEL)/dundee.gcg13.linetrees $(SCRIPTS)/deleteLeaves.py | $(SRCMODEL)
	cat $(word 1, $^) | python $(word 2, $^) > $@

.PRECIOUS: %dundee.gold.linetrees
%dundee.gold.linetrees: $(SRCMODEL)/dundee.stripped.linetrees genmodel/dundee.sents \
			$(SCRIPTS)/plugDundee.py | genmodel
	cat $(word 2, $^) | sed 's/garbage\/sentimental/garbage \/ sentimental/g' \
	| sed 's/Oxford-educated/Oxford- educated/g' \
	| sed 's/4'\''s/4 '\''s/g' \
	| sed 's/'\''s-find-a-cure-fast/'\''s- find- a- cure- fast/g' \
	| sed 's/\([Aa]\)lcohol-related/\1lcohol- related/g' \
	| sed 's/and-mouth/and- mouth/g' \
	| sed 's/fork-lift/fork- lift/g' \
	| sed 's/in-the-box/in- the- box/g' \
	| sed 's/potato-and/potato- and/g' \
	| sed 's/Faction\/Dogs/Faction \/ Dogs/g' \
	| sed 's/three- year-old/three- year- old/g' \
	> $(word 2, $^).tmp
	python $(word 3, $^) $(word 1, $^) $(word 2, $^).tmp > $@
	rm $(word 2, $^).tmp
    
.PRECIOUS: %dundee.gold.edit.editabletrees
%dundee.gold.edit.editabletrees: %dundee.gold.linetrees bin/indent
	cat $(word 1, $^) | bin/indent > $@

.PRECIOUS: %.fg.linetrees
%.fg.linetrees:  %.linetrees  ../dundee/scripts/annotateFGTrans.pl
	cat $(word 1,$^)  |  perl $(word 2,$^)  >  $@

.PRECIOUS: %.fg.t.linetrees
%.fg.t.linetrees:  %.linetrees  ../dundee/scripts/annotateFGTrans.pl
	cat $(word 1,$^)  |  perl $(word 2,$^) -t >  $@
    
.PRECIOUS: %.semlctrans.linetrees
%.semlctrans.linetrees: ../incrsem/scripts/semlctrans.pl %.linetrees
	@ echo "Executing semantic left-corner transform"
	cat $(word 2,$^) | perl $(word 1,$^) > $@
    
.PRECIOUS: %.depth.linetrees
%.depth.linetrees: ../gcg/scripts/annotateDepth.py %.linetrees
	@ echo "Annotating the gold standard corpus for center-embedding depth."
	cat $(word 2, $^) | python $(word 1, $^) > $@
    
.PRECIOUS: %.embddata
%.embddata: %.linetrees ../resource-dundee/scripts/calccenterembeddingtoks.py
	@ echo "Generating gold standard center embedding predictors by token."
	python $(word 2, $^) $(word 1, $^) > $@
    
.PRECIOUS: %.dltdata
%.dltdata: %.linetrees ../resource-dundee/scripts/calcDLTtoks.py
	@ echo "Generating gold standard DLT values by token."
	python $(word 2, $^) $(word 1, $^) > $@
    
.PRECIOUS: golddata/dundee.gold.golddata
golddata/dundee.gold.golddata: golddata/dundee.gold.dltdata \
	golddata/dundee.gold.semlctrans.fg.t.depth.embddata
	paste -d" " $(word 1, $^) <(cut -d" "    -f3- $(word 2, $^)) > $@

.PRECIOUS: %.goldrules
%.goldrules: %.linetrees ../general/scripts/editabletrees2linetrees.pl ../general/scripts/trees2rules.pl
	cat $(word 1, $^) | perl $(word 2, $^) | perl $(word 3, $^) | sort | uniq -c > $@