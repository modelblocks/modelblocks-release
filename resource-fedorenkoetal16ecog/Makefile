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
Fedorenkoetal16ECoG:
  kernel: ECoGCorpus
  text: Fedorenko et al 16 (ECoG corpus)
  value: fedorenkoetal16
endef

FEDECOG-BASENAME := genmodel/fedorenkoetal16ecog_sents.wsj02to21-GRAMMAR-nol-prtrm-3sm-synproc-+c_+u_+b5000.gcg15_nodashtags.merged.syn.5-kenlm

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
CONFIG := $(CONFIGDIR)/user-fedorenkoetal16ecog-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
FEDECOGDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(FEDECOGDIR))))
RESOURCE-NAME := Fedorenko et al 2016 ECoG corpus
define RESOURCE-DESCR =

NAME: Fedorenko et al 2016 ECoG corpus
MB POINTER FILE: config/user-fedorenkoetal16ecog-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: A corpus of ecog measures of gamma power from subjects in RSVP experiment.
endef
FEDECOG-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-FEDECOG = $(error $(NATSTORIES-ERROR))
endif
endif



%/user-fedorenkoetal16ecog-directory.txt: | %   
	echo '/home/corpora/original/english/fedorenkoetal16_ecog' > $@

################################################################################
#
#  Data formatting items
#
################################################################################

fedorenkoetal16ecog_sent%.linetoks: $$(DEP-FEDECOG) $$(DEP-TOKENIZER) $(CONFIG) $$(FEDECOGDIR)/materials_visual/1_sent/$$*.txt $$(TOKENIZERDIR)/ptb_tokenizer.sed
	tr '\n' ' ' < $(word 2, $^) | tr '[:upper:]' '[:lower:]' | sed -f $(word 3, $^) | paste -d'\0' - <(echo ".") | sed 's/./\u&/' > $@

%/fedorenkoetal16ecog_sents.linetoks: $$(DEP-FEDECOG) $(CONFIG) $$(foreach file,$$(wildcard $$(FEDECOGDIR)/materials_visual/1_sent/*.txt),%/fedorenkoetal16ecog_sent$$(notdir $$(basename $$(file))).linetoks) | genmodel
	awk '{print $$0}' $(sort $(wordlist 2, 200, $^)) > $@

%/fedorenkoetal16ecog_sents.lineitems: $$(DEP-TOKENIZER) %/fedorenkoetal16ecog_sents.linetoks $$(TOKENIZERDIR)/ptb_untokenizer.sed
	cat $(word 1, $^) | sed -f $(word 2, $^) > $@

