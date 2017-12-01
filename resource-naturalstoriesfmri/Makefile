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

define ParamVal
NaturalStoriesFMRI:
  kernel: fMRICorpus
  text: Natural Stories (fMRI corpus)
  value: naturalstoriesFMRI
endef

.SUFFIXES:
.SECONDEXPANSION:

VPATH += $(RESOURCE-NATSTORIESFMRI)

NATSTOR-STORIES := Boar Aqua MatchstickSeller KingOfBirds Elvis MrSticky HighSchool Roswell Tulips Tourettes
NATSTOR-ROI := LPostTemp LPostTempZ LAntTemp LAntTempZ LIFG LIFGZ LIFGorb LIFGorbZ LMFG LMFGZ LAngG LAngGZ LHip RHip Hip LHipZ RHipZ HipZ
NATSTORIESFMRI-BASENAME := genmodel/naturalstoriesfmri.wsj02to21-gcg15-decoupled-fg-3sm-bd-x+efabp-+c_+b5000.syn.5-kenlm.hrf
NATSTORIESFMRI-LMEDEFAULT := -XIvw
$(eval $(call FMRI-HRF-CONTROLS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),fmriEmpty,$(NATSTORIESFMRI-LMEDEFAULT)))
$(eval $(call FMRI-HRF-EXPERIMENTS,naturalstoriesfmri,$(NATSTORIESFMRI-BASENAME),$(NATSTOR-ROI),nsfMRI,$(NATSTORIESFMRI-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-naturalstories-fmri-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORFMRIDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORFMRIDIR))))
RESOURCE-NAME := The Natural Stories Corpus (fMRI)
define RESOURCE-DESCR =

NAME: The Natural Stories Corpus (fMRI)
MB POINTER FILE: config/user-naturalstories-fmri-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: fMRI data collected from acoustic presentation of the
Natural Stories corpus
endef
NATSTORIESFMRI-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-NATSTORIESFMRI = $(error $(NATSTORIESFMRI-ERROR))
endif
endif



%/user-naturalstories-fmri-directory.txt: | %   
	echo '/home/corpora/original/english/naturalstories_fmri' > $@

################################################################################
#
#  Recipes
#
################################################################################

%naturalstoriesfmri.linetoks: %naturalstories.linetoks
	cp $(word 1, $^) $@

%naturalstoriesfmri.lineitems: %naturalstories.lineitems
	cp $(word 1, $^) $@

naturalstoriesfmri%.tokmeasures: naturalstories%.tokmeasures
	cp $(word 1, $^) $@

naturalstoriesfmri%.itemmeasures: naturalstories%.itemmeasures
	cp $(word 1, $^) $@


%naturalstoriesfmri.evmeasures: $$(DEP-NATSTORIESFMRI) $(NATSTORFMRIDIR)/naturalstoriesFMRI.evmeasures $(NATSTORIESFMRI-SCRIPTS)/preprocess_natstor_fmri.py \
%naturalstories.soundPower.itemmeasures $(RT-SCRIPTS)/merge_tables.py $(CONFIGDIR)/user-naturalstories-fmri-directory.txt | %
	python $(word 4, $^) $(word 3, $^) <(cat $(word 1, $^) | python $(word 2, $^) $(NATSTOR-STORIES)) sampleid > $@
