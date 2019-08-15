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
#  include $(RESOURCE-NATSTORIES)/Makefile
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
CONFIG := $(CONFIGDIR)/user-conlen-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CONLENDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CONLENDIR))))
RESOURCE-NAME := The constituent length fMRI corpus
define RESOURCE-DESCR =

NAME: The constituent length fMRI corpus
MB POINTER FILE: config/user-conlen-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: Materials used to study effects of constituent length on brain responses
endef
CONLEN-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-CONLEN = $(error $(CONLEN-ERROR))
endif
endif



%/user-conlen-directory.txt: | %   
	echo '/home/corpora/original/english/constituent_length' > $@

################################################################################
#
#  Recipes
#
################################################################################

%conlen.itemmeasures: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length.itemmeasures $(CONFIGDIR)/user-conlen-directory.txt | %
	cp $< $@

conlen%.lineitems: $$(DEP-CONLEN) $(CONLENDIR)/constituent_length_$$*.lineitems | genmodel
	cp $< $@

conlen%.linetoks: $$(DEP-TOKENIZER) conlen%.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g' | sed 's/\r //g' > $@

%conlenc.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees %conlenc.linetoks
	python $^ > $@

%conlenc.12.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 30 $^ > $@

%conlenc.6.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 90 $^ | tail -n 60 > $@

%conlenc.4.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 180 $^ | tail -n 90 > $@

%conlenc.3.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 300 $^ | tail -n 120 > $@

%conlenc.2.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	head -n 480 $^ | tail -n 180 > $@

%conlenc.1.stripped.linetrees: $(RESOURCE-CONLEN)/srcmodel/conlenc.stripped.linetrees
	tail -n 360 $^ > $@

%conlenjab.stripped.linetrees: %conlenc.12.stripped.linetrees %conlenc.4.stripped.linetrees %conlenc.1.stripped.linetrees
	cat $^ > $@

%conlenjab.linetrees: $(GCG-SCRIPTS)/plugLeaves.py %conlenjab.stripped.linetrees %conlenjab.linetoks
	python $^ > $@

