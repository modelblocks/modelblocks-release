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
#  include $(RESOURCE-TOKENIZER)/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-RT)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:


################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-bk21cloze-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BK21DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(UCLDIR))))
RESOURCE-NAME := BK21 cloze SPR dataset (Brothers and Kuperberg, 2021)
define RESOURCE-DESCR =

NAME: BK21 cloze SPR dataset (Brothers and Kuperberg, 2021)
MB POINTER FILE: config/user-bk21cloze-directory.txt
AVAILABILITY: Free
URL: https://osf.io/b9kns/
DESCRIPTION: Self-paced-reading data from subjects reading the same critical word in
experimentally-manipulated contexts that make more predictable or less predictable.
endef
BK21-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-BK21 = $(error $(BK21-ERROR))
endif
endif


%/user-bk21cloze-directory.txt: | %
	echo '/fs/project/lin-dept/corpora/original/english/bk21_cloze' > $@


################################################################################
#
#  SPR data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(BK21-SCRIPTS)/%.ini
	cp $< $@

%bk21cloze.evmeasures: $(BK21DIR)/SPRT_LogLin_216.csv $(BK21-SCRIPTS)/process_bk21cloze.py
	python3 $(word 2,$^) $(word 1,$^) > $@

# %ucl.evmeasures: $$(DEP-UCL) \
# $(CONFIGDIR)/user-ucl-directory.txt \
# $(UCL-SCRIPTS)/build_ucl_corpus.py \
# $(UCLDIR)/eye_movement/eyetracking.RT.txt \
# $(RT-SCRIPTS)/rename_cols.py \
# $(RT-SCRIPTS)/filter_cols.awk \
# $(CDR-SCRIPTS)/compute_timestamps_from_RT.py | %
# 	cat $(word 3, $^) | python $(word 2,$^) > $@.tmp
# 	paste -d' ' $@.tmp <(awk -f $(word 5, $^) -v cols=sentid $@.tmp | python $(word 4, $^) sentid docid) | python $(word 6, $^) > $@
# 	rm $@.tmp

################################################################################
#
#  Sents files
#
################################################################################

%bk21cloze.sentitems: $(BK21-SCRIPTS)/bk21cloze.txt
	cp $< $@

%bk21cloze.itemmeasures: %bk21cloze.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  python3 $(word 2, $^) >  $@
