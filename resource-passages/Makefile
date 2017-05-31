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

define ParamVal
NaturalStories:
  kernel: SPRCorpus
  text: Natural Stories (self-paced reading corpus)
  value: naturalstories
endef

.SUFFIXES:
.SECONDEXPANSION:

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-passages-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
PASSAGESDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(PASSAGESDIR))))
RESOURCE-NAME := The MIT Sentence Passages corpus
define RESOURCE-DESCR =

NAME: The MIT Sentence Passages corpus
AVAILABILITY: Unreleased
DESCRIPTION: A corpus of fMRI bold responses by subjects to audio presentation
of short passages (3-4 sentences each) in isolation.
endef
PASSAGES-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-PASSAGES = $(error $(PASSAGES-ERROR))
endif
endif



%/user-passages-directory.txt: | %   
	echo '/home/corpora/original/english/MITpassages' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

genmodel/%.xlsx: $$(DEP-PASSAGES) $(PASSAGESDIR)/$$*.xlsx | genmodel
	cp $(word 1, $^) $@

%passages72.lineitems: genmodel/passages_eval.fromxlsx.csv | genmodel
	head -n+73 $(word 1, $^) | tail -n+2 | \
	awk '{for (i=3; i<7; i++) if ($$i != "") print $$i}' FPAT="([^ ]*)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' > $@

%passages72.linetoks: $$(DEP-TOKENIZER) %passages72.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $(word 1, $^) | $(word 2, $^) > $@

%passages96.lineitems: genmodel/passages_96concepts.fromxlsx.csv | genmodel
	tail -n+2 $(word 1, $^) | awk '{print $$3}' FPAT="([^ ]+)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' > $@

%passages96.linetoks: $$(DEP-TOKENIZER) %passages96.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt
	cat $(word 1, $^) | $(word 2, $^) > $@

