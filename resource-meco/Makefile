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

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-DUNDEE)

MECOLANGS = du ee en fi ge gr he it ko no ru sp tr


################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-meco-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
MECODIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(MECODIR))))
RESOURCE-NAME := The multilingual eye-movement corpus (MECO)
define RESOURCE-DESCR =

NAME: The multilingual eye-movement corpus
MB POINTER FILE: config/user-meco-directory.txt
AVAILABILITY: https://osf.io/3527a/
DESCRIPTION: A multilingual corpus of eye-tracking measures from subjects who read
12 stories (Siegelman et al., 2022).
endef
MECO-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-MECO = $(error $(MECO-ERROR))
endif
endif


%/user-meco-directory.txt: | %
	echo '/fs/project/schuler.77/meco' > $@


################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
define build_meco_part_params
%$(1)meco.part_params.ini: $(MECO-SCRIPTS)/meco.part_params.ini
	cp $$< $$@
endef

define build_meco_evmeasures
%$(1)meco.evmeasures: $(MECODIR)/joint_fix_trimmed.rda $(MECODIR)/joint_l1_acc_breakdown.rda $(MECO-SCRIPTS)/preprocess_meco.R $(MECO-SCRIPTS)/process_meco.py %$(1)meco.sentitems
	Rscript $$(word 3,$$^) $$(word 1,$$^) $$(word 2,$$^) $(1) | python3 $$(word 4,$$^) $$(word 5,$$^) > $$@
endef


################################################################################
#
#  Sents files
#
################################################################################

define build_meco_sentitems
.PRECIOUS: %$(1)meco.sentitems
%$(1)meco.sentitems: $(MECODIR)/joint_fix_trimmed.rda $(MECO-SCRIPTS)/build_meco_sentitems.R
	Rscript $$(word 2,$$^) $$(word 1,$$^) $(1) > $$@
endef

%meco.itemmeasures: %meco.sentitems $(MECO-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  python3 $(word 2, $^) >  $@

$(foreach i,$(MECOLANGS),\
	$(eval $(call build_meco_sentitems,$(i)))\
	$(eval $(call build_meco_evmeasures,$(i)))\
	$(eval $(call build_meco_part_params,$(i)))\
)
