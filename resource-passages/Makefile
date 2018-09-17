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
Passages72:
  kernel: fMRICorpus
  text: MIT Passages n72 (fMRI corpus)
  value: passages72
endef

define ParamVal
Passages96:
  kernel: fMRICorpus
  text: MIT Passages n96 (fMRI corpus)
  value: passages96
endef

.SUFFIXES:
.SECONDEXPANSION:

PASSAGES-ROI := LPostTemp LAntTemp LAngG LIFG LMFG LIFGorb AllROI
PASSAGES72-BASENAME := genmodel/passages72.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000.syn.5-kenlm.bysent
PASSAGES72-LMEDEFAULT := -vw 
$(eval $(call FMRI-BYSENT-EXPERIMENTS,passages72,$(PASSAGES72-BASENAME),$(PASSAGES-ROI),passagesBasic,$(PASSAGES72-LMEDEFAULT)))
PASSAGES96-BASENAME := genmodel/passages96.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000.syn.5-kenlm.bysent
PASSAGES96-LMEDEFAULT := -vw
$(eval $(call FMRI-BYSENT-EXPERIMENTS,passages96,$(PASSAGES96-BASENAME),$(PASSAGES-ROI),passagesBasic,$(PASSAGES96-LMEDEFAULT)))
PASSAGES-BASENAME := genmodel/passages.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000.syn.5-kenlm.bysent
PASSAGES-LMEDEFAULT := -vw
$(eval $(call FMRI-BYSENT-EXPERIMENTS,passages,$(PASSAGES-BASENAME),$(PASSAGES-ROI),fmriEmpty,$(PASSAGES-LMEDEFAULT)))
$(eval $(call FMRI-BYSENT-EXPERIMENTS,passages,$(PASSAGES-BASENAME),$(PASSAGES-ROI),nwrds,$(PASSAGES-LMEDEFAULT)))
$(eval $(call FMRI-BYSENT-EXPERIMENTS,passages,$(PASSAGES-BASENAME),$(PASSAGES-ROI),nwrdswlen,$(PASSAGES-LMEDEFAULT)))

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
MB POINTER FILE: config/user-passages-directory.txt
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
#  Data formatting items
#
################################################################################

genmodel/%.xlsx: $$(DEP-PASSAGES) $(PASSAGESDIR)/$$*.xlsx $(CONFIGDIR)/user-passages-directory.txt | genmodel
	cp $(word 1, $^) $@

%passages72.lineitems: genmodel/passages_eval.fromxlsx.csv | genmodel
	head -n+73 $(word 1, $^) | tail -n+2 | \
	awk '{for (i=3; i<7; i++) if ($$i != "") print $$i}' FPAT="([^ ]*)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' | sed 's/ $$//g' > $@

genmodel/passages72.linetoks: $$(DEP-TOKENIZER) genmodel/passages72.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $(word 1, $^) | $(word 2, $^) | sed 's/ $$//g' > $@

%passages96.lineitems: genmodel/passages_96concepts.fromxlsx.csv | genmodel
	tail -n+2 $(word 1, $^) | awk '{print $$3}' FPAT="([^ ]+)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' | sed 's/ $$//g' > $@

genmodel/passages96.linetoks: $$(DEP-TOKENIZER) genmodel/passages96.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $(word 1, $^) | $(word 2, $^) | sed 's/ $$//g' > $@

passages%.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-PASSAGES)/srcmodel/passages$$*.stripped.linetrees passages%.linetoks
	python $^ > $@

genmodel/passages.linetoks: genmodel/passages72.linetoks genmodel/passages96.linetoks
	cat $(word 1, $^) $(word 2, $^) > $@

genmodel/passages.lineitems: genmodel/passages72.lineitems genmodel/passages96.lineitems	
	cat $(word 1, $^) $(word 2, $^) > $@

genmodel/passages.linetrees: genmodel/passages72.linetrees genmodel/passages96.linetrees
	cat $(word 1, $^) $(word 2, $^) > $@

genmodel/passages.evmeasures: genmodel/passages72.evmeasures genmodel/passages96.evmeasures $(PASSAGES-SCRIPTS)/cumSentID.py
	cat $(word 1, $^) $(word 2, $^) | python $(word 3, $^) > $@

passages%.evmeasures: $$(DEP-PASSAGES) $(PASSAGESDIR)/passages_data_n$$*.csv \
$(RT-SCRIPTS)/rename_cols.py \
$(PASSAGES-SCRIPTS)/cumSentID.py \
$(FMRI-SCRIPTS)/averageROI.py \
$(CONFIGDIR)/user-passages-directory.txt | genmodel
	cat $(word 1, $^) | sed 's/,/ /g;s/_sent/ /g;s/passage//g;s/con_name/docid sentid/g' | \
	python $(word 2, $^) subj_name subject con_name docid psc_LPostTemp boldLPostTemp psc_LAntTemp boldLAntTemp psc_LAngG boldLAngG psc_LIFG boldLIFG psc_LMFG boldLMFG psc_LIFGorb boldLIFGorb | \
	awk 'NR == 1 {print;}; NR > 1 {$$3 = $$3 - 1; print;};' | \
	python $(word 3, $^) | \
	python $(word 4, $^) $(PASSAGES-ROI) > $@

passages%-devbatch: $(foreach roi, $(PASSAGES-ROI), $(foreach pred, $(BYSENT-PREDS), $$(PASSAGES$$*-BASENAME).$(roi)passagesBasic.$(pred).$$(PASSAGES$$*-LMEDEFAULT)..lrt));

passages-devbatch: $(foreach roi, $(PASSAGES-ROI), $(foreach pred, $(BYSENT-PREDS), $$(PASSAGES-BASENAME).$(roi)passagesBasic.$(pred).$$(PASSAGES-LMEDEFAULT)..lrt));

