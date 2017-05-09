
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
#  Includes
#
#  Include statements to this file should be preceded by the following includes
#
#  include $(dir $(CURDIR))/general/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-BMMM)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
###############################################################################

.SUFFIXES:
.SECONDEXPANSION:

NAT := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-dmv-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
DMVDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(DMVDIR))))
endif
RESOURCE-NAME := The Dependency Model with Valence (DMV)
define RESOURCE-DESCR =

NAME: The Dependency Model with Valence (DMV)
AVAILABILITY: Free
URL: https://code.google.com/archive/p/pr-toolkit/
DESCRIPTION: An implementation (Gillenwater et al 2010) of the Dependency
Model with Valence parser (Klein & Manning 2004) for unsupervised
dependency parsing.
endef
DMV-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-DMV = $(error $(DMV-ERROR))
endif



%/user-dmv-directory.txt: | %
	echo '/home/compling/pr-dep-parsing.WILS' > $@

################################################################################
#
#  Recipes
#
################################################################################

.PRECIOUS: %.1.dmv.conll %.1.u.dmv.conll
%.1.dmv.conll %.1.u.dmv.conll: $$(DEP-DMV) %.bmmmbase.conll $(DMVDIR) $(CONFIGDIR)/user-dmv-directory.txt $(CONFIGDIR)/user-javaflags.txt
	echo "name=Conll" > $*.1.dmv.params
	echo "lowercase=false" >> $*.1.dmv.params
	echo "transitive=false" >> $*.1.dmv.params
	echo "reader-type=naacl2012-fine" >> $*.1.dmv.params
	echo "train-name=dmv_train" >> $*.1.dmv.params
	echo "train-file=$(abspath $(word 1, $^).temp)" >> $*.1.dmv.params
	echo "devname=dmv_devel" >> $*.1.dmv.params
	echo "dev-file=$(abspath $(word 1, $^).temp)" >> $*.1.dmv.params
	echo "test-name1=dmv_test" >> $*.1.dmv.params
	echo "test-file1=$(abspath $(word 1, $^).temp)" >> $*.1.dmv.params
	cp $(word 1, $^) $(word 1, $^).temp
	cd $(word 2, $^); java $(shell cat $(word 4, $^)) -cp '$(word 2, $^)/*' depparsing.programs.RunModel -corpus-params $(abspath $*.1.dmv.params) -model-init K_AND_M -dvalency 2 -cvalency 1 -child-backoff 0.66 -num-em-iters 20 -trainingType 0 -stats-file $(word 2, $^)/stats-example.txt -output-prefix $(abspath $(dir $*))/
	cat $(dir $*)dmv_test | awk ' { t = $$7; $$7 = $$8; $$8 = t; print; } ' OFS=$$'\t' | sed 's/\t\t\+//g' > $@
	rm $(dir $*)dmv_test

.PRECIOUS: %.dmv.conll
%.dmv.conll: $$(DEP-DMV) %.bmmm.conll $(DMVDIR) $(CONFIGDIR)/user-dmv-directory.txt $(CONFIGDIR)/user-javaflags.txt
	echo "name=Conll" > $*.dmv.params
	echo "lowercase=false" >> $*.dmv.params
	echo "transitive=false" >> $*.dmv.params
	echo "reader-type=naacl2012-fine" >> $*.dmv.params
	echo "train-name=dmv_train" >> $*.dmv.params
	echo "train-file=$(abspath $(word 1, $^).temp)" >> $*.dmv.params
	echo "devname=dmv_devel" >> $*.dmv.params
	echo "dev-file=$(abspath $(word 1, $^).temp)" >> $*.dmv.params
	echo "test-name1=dmv_test" >> $*.dmv.params
	echo "test-file1=$(abspath $(word 1, $^).temp)" >> $*.dmv.params
	cp $(word 1, $^) $(word 1, $^).temp
	cd $(word 2, $^); java $(shell cat $(word 4, $^)) -cp '$(word 2, $^)/*' depparsing.programs.RunModel -corpus-params $(abspath $*.dmv.params) -model-init K_AND_M -dvalency 2 -cvalency 1 -child-backoff 0.66 -num-em-iters 20 -trainingType 0 -stats-file $(word 2, $^)/stats-example.txt -output-prefix $(abspath $(dir $*))/
	cat $(dir $*)dmv_test | awk ' { t = $$7; $$7 = $$8; $$8 = t; print; } ' OFS=$$'\t' | sed 's/\t\t\+//g' > $@
	rm $(dir $*)dmv_test

%.dmv.linetrees: %.dmv.conll $(LTREES-SCRIPTS)/deps2trees.py
	cat $< | python $(word 2, $^) > $@

# Generates a space-delimited table of accuracy measures by tagset size
# for each sample file in the user-supplied project directory
# 
# Target stem template:
#     
#     <gold-base>.<target-base>-<iters>.<start>.<end>.nTag.dmv.itermeasures
#
# where the gold and target bases are the linetrees specifications to be used in the
# syneval, with periods replaced with hyphens and the number of iterations specified
# by <iters>. <start> is the smallest tagset size, and <end> is the largest. Only
# tagsets <= 100 classes are supported.
#
.PRECIOUS: %.tags.dmv.itermeasures
%.tags.dmv.itermeasures: ../resource-grammar-induc/scripts/extract_syneval_table.py $$(foreach n, $$(wordlist $$(subst .,,$$(suffix $$(basename $$*))), $$(subst .,,$$(suffix $$*)),$$(NAT)), $$(basename $$(basename $$(basename %))).$$(subst .,-,$$(basename $$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))$$(subst)-$$(n)-$$(subst .,,$$(suffix $$(subst -,.,$$(basename $$(basename $$*)))))-dmv.induc.diffname.syneval);
	python $< -x tags -s $(subst .,,$(suffix $(basename $*))) $(wordlist 2, $(words $^), $^) > $@

# Generates a space-delimited table of accuracy measures by iteration
# for each sample file in the user-supplied project directory
# 
# Target stem template:
#     
#     <gold-base>.<target-base>-<tags>.<start>.<end>.nTag.dmv.itermeasures
#
# where the gold and target bases are the linetrees specifications to be used in the
# syneval, with periods replaced with hyphens and the number of PoS tags specified
# by <tags>. <start> is the smallest number of iterations, and <end> is the largest.
# Only iterations counts <= 100 are supported.
#
.PRECIOUS: %.iter.dmv.itermeasures
%.iter.dmv.itermeasures: ../resource-grammar-induc/scripts/extract_syneval_table.py $$(foreach n, $$(wordlist $$(subst .,,$$(suffix $$(basename $$*))), $$(subst .,,$$(suffix $$*)),$$(NAT)), $$(basename $$(basename $$(basename %))).$$(suffix $$(basename $$(basename $$*)))-$$(n)-dmv.induc.diffname.syneval);
	python $< -x iter -s $(subst .,,$(suffix $(basename $*))) $(wordlist 2, $(words $^), $^) > $@


