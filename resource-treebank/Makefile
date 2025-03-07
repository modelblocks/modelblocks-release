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
#  Includes
#
#  Include statements to this file should be preceded by the following includes
#
#  include $(dir $(CURDIR))resource-general/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

BROWNTRAINSECTS = cf cg ck cl cm cn cp cr

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-treebank-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
TREEBANKDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(TREEBANKDIR))))
RESOURCE-NAME := The Penn Treebank (PTB)
define RESOURCE-DESCR =

NAME: The Penn Treebank (PTB)
MB POINTER FILE: config/user-treebank-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/ldc99t42
DESCRIPTION: One million words of 1989 Wall Street Journal material annotated in Treebank II style.
A small sample of ATIS-3 material annotated in Treebank II style.
Switchboard tagged, dysfluency-annotated, and parsed text.
A fully tagged version of the Brown Corpus.
Brown parsed text.
endef
TREEBANK-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-TREEBANK = $(error $(TREEBANK-ERROR))
endif
endif


%/user-treebank-directory.txt: | %   
	echo '/fs/project/lin-dept/corpora/original/english/penn_treebank_3' > $@
%/user-treebank-vadas-directory.txt: | %
	echo '/fs/project/lin-dept/corpora/original/english/penn_treebank_3_vadas_np_bracketing' > $@

################################################################################
#
#  Recipes
#
################################################################################
    
genmodel/brown%.senttrees: $$(DEP-TREEBANK) $(CONFIGDIR)/user-treebank-directory.txt $(TREEBANKDIR)/parsed/mrg/brown/$$* $(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	touch $@
	rm $@
	for f in $(word 2,$^)/*.mrg; do echo '!ARTICLE' >> $@; cat $$f | sed 's/\[/-LSB-/g;s/\]/-RSB-/g;' | perl $(word 3,$^) >> $@; done
#	echo $^
#	cat $(word 2,$^)/*.mrg | sed 's/\[/-LSB-/g;s/\]/-RSB-/g;' | perl $(word 3,$^) > $@

genmodel/brownTRAIN.senttrees: $(foreach sect,$(BROWNTRAINSECTS),genmodel/brown$(sect).senttrees) | genmodel
	cat $^ > $@

genmodel/wsj%.senttrees: $$(DEP-TREEBANK) $(CONFIGDIR)/user-treebank-directory.txt $(TREEBANKDIR)/parsed/mrg/wsj/$$* \
$(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	touch $@
	rm $@
	for f in $(word 2,$^)/*.mrg; do echo '!ARTICLE' >> $@; cat $$f | perl $(word 3,$^) >> $@; done
#	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@
genmodel/wsjv%.linetrees: $$(DEP-TREEBANK) $(CONFIGDIR)/user-treebank-vadas-directory.txt $$(shell cat $(CONFIGDIR)/user-treebank-vadas-directory.txt)/parsed/mrg/wsj/$$* \
$(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@

%/wsj02to21.senttrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,%/wsj$(sect).senttrees)
	cat $^ > $@
%/wsj00to24.linetrees:  $(foreach sect,00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24,%/wsj$(sect).linetrees)
	cat $^ > $@
%/wsjv02to21.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,%/wsjv$(sect).linetrees)
	cat $^ > $@

genmodel/swbd%.linetrees: $$(DEP-TREEBANK) $(CONFIGDIR)/user-treebank-directory.txt $(TREEBANKDIR)/parsed/mrg/swbd/$$* \
$(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	cat $(word 2, $^)/*.mrg | perl $(word 3, $^) | awk '/^\s*\(/' > $@

%/swbd2to4.linetrees: $(foreach sect, 2 3 4, %/swbd$(sect).linetrees)
	cat $^ > $@

## Empty coref file for linked modifier, for generic casp item (gcg15.long.linked.morphed)...
genmodel/brown%.nocoref:
	touch $@
genmodel/wsj%.nocoref:
	touch $@



