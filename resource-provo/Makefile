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
CONFIG := $(CONFIGDIR)/user-provo-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
PROVODIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(PROVODIR))))
RESOURCE-NAME := The Provo eye-tracking corpus
define RESOURCE-DESCR =

NAME: The Provo eye-tracking corpus
MB POINTER FILE: config/user-provo-directory.txt
AVAILABILITY: Public
DESCRIPTION: A corpus of eye-tracking measures from Luke & Christianson (2016)
endef
PROVO-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-PROVO = $(error $(PROVO-ERROR))
endif
endif


    
%/user-provo-directory.txt: | %
	echo '/fs/project/lin-dept/corpora/original/english/provo' > $@
    
################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(PROVO-SCRIPTS)/%.ini
	cp $< $@

%/provo.evmeasures: $$(DEP-PROVO) $(PROVO-SCRIPTS)/merge_provo.py $(PROVODIR)/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv $(PROVO-SCRIPTS)/process_provo.py \
%/provo.itemmeasures | %
	python3 $< <(cat $(word 2, $^) | python3 $(word 3,$^)) $(word 4, $^) > $@


################################################################################
#
#  Sents files
#
################################################################################    

%/provo.itemmeasures: $$(DEP-PROVO) $(PROVODIR)/Provo_Corpus-Predictability_Norms.csv $(PROVO-SCRIPTS)/get_provo_itemmeasures.py
	cat $< | python3 $(word 2, $^) > $@

%/provo.sentitems: %/provo.itemmeasures $(PROVO-SCRIPTS)/get_provo_sentitems.py
	cat $< | python3 $(word 2, $^) > $@

%/provo.senttoks: %/provo.sentitems $(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $< | $(word 2, $^) | perl -pe 's/ +$$//g' > $@

%provo.casp.notoks.senttrees:  $(RESOURCE-PROVO)/srcmodel/provo.stripped.senttrees  $(RESOURCE-GCG)/scripts/lengthentrees.py
	cat $<  |  python3 $(word 2,$^)  >  $@

