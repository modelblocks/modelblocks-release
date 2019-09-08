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
CONFIG := $(CONFIGDIR)/user-tom-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
TOMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(TOMDIR))))
RESOURCE-NAME := The Saxe Theory of Mind corpus
define RESOURCE-DESCR =

NAME: The Saxe Theory of Mind corpus
MB POINTER FILE: config/user-tom-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: Materials used by Dodell-Feder et al., (2010)
endef
TOM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-TOM = $(error $(TOM-ERROR))
endif
endif



%/user-tom-directory.txt: | %   
	echo '/home/corpora/original/english/saxe_tom' > $@

################################################################################
#
#  Recipes
#
################################################################################

%tom.itemmeasures: $$(DEP-TOM) $(TOMDIR)/tom.itemmeasures $(CONFIGDIR)/user-tom-directory.txt | %
	cp $< $@

genmodel/tom.lineitems: genmodel/tom.itemmeasures $(GENERAL-SCRIPTS)/itemmeasures2lineitems.py | genmodel
	cat $< | python $(word 2, $^) > $@

%tom.linetoks: $$(DEP-TOKENIZER) %tom.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $< | $(word 2, $^) | sed 's/ $$//g'> $@

%tom.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-TOM)/srcmodel/tom.stripped.linetrees %tom.linetoks
	python $^ > $@

