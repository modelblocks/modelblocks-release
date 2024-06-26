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
CONFIG := $(CONFIGDIR)/user-williams-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
WILLIAMSDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(WILLIAMSDIR))))
RESOURCE-NAME := The Williams intracranial recordings corpus
define RESOURCE-DESCR =

NAME: The Williams intracranial recordings corpus
MB POINTER FILE: config/user-williams-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: Materials used to stimulate neural responses to language in different conditions
endef
WILLIAMS-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-WILLIAMS = $(error $(WILLIAMS-ERROR))
endif
endif



%/user-williams-directory.txt: | %   
	echo '/fs/project/schuler.77/corpora/original/english/williams' > $@

################################################################################
#
#  Recipes
#
################################################################################

%williams.itemmeasures: $$(DEP-WILLIAMS) $(WILLIAMSDIR)/williams.itemmeasures $(CONFIGDIR)/user-williams-directory.txt | %
	cp $< $@

%williams.lineitems: %williams.itemmeasures $(GENERAL-SCRIPTS)/itemmeasures2lineitems.py
	cat $< | python $(word 2, $^) > $@

%williams.linetoks: $$(DEP-TOKENIZER) %williams.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g'> $@

%williams.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-WILLIAMS)/srcmodel/williams.stripped.linetrees %williams.linetoks
	python $^ > $@

