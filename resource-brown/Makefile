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
CONFIG := $(CONFIGDIR)/user-brown-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BROWNDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORMAZEDIR))))
RESOURCE-NAME := The Brown SPR Corpus
define RESOURCE-DESCR =

NAME: The Brown SPR Corpus
MB POINTER FILE: config/user-brown-directory.txt
AVAILABILITY: https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8
DESCRIPTION: The SPR data of Smith & Levy 2013
endef
BROWN-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-BROWN = $(error $(BROWN-ERROR))
endif
endif

%/user-brown-directory.txt: | %   
	echo '/fs/project/lin-dept/corpora/original/english/SPR_brown' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(BROWN-SCRIPTS)/%.ini  |  scripts
	cp $< $@

%brown.evmeasures: $$(DEP-BROWN) $(BROWNDIR)/corpora/brown_spr.csv $(BROWN-SCRIPTS)/process_brown.py %brown.itemmeasures
	cat $< | python3 $(word 2, $^) $(word 3, $^) > $@


################################################################################
#
#  Sents files
#
################################################################################    

brown%itemmeasures: $$(DEP-BROWN) $(BROWNDIR)/corpora/bnc_brown.txt $(BROWN-SCRIPTS)/get_brown_itemmeasures.py $(BROWNDIR)/human_rts/bnc-brown/bnc-brown.csv \
$(BROWNDIR)/corpora/brown_spr.csv | %
	cat $< | python3 $(word 2, $^) $(word 3, $^) $(word 4, $^) > $@

%brown.sentitems: %brown.itemmeasures $(BROWN-SCRIPTS)/get_brown_sentitems.py | %
	cat $< | python3 $(word 2, $^) > $@

%brown.senttoks: %brown.sentitems $(TOKENIZERDIR)/ptb_tokenizer.sed | %
	cat $< | $(word 2, $^) | perl -pe 's/ +$$//g' > $@
