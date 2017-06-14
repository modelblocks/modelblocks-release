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

VPATH += $(RESOURCE-PASSAGES)

PASSAGES-ROI := LPostTemp LAntTemp LAngG LIFG LMFG LIFGorb
PASSAGES-PREDS := nwrds fjprob noFprob totsurp lexsurp synsurp fwprob5surp wlen
PASSAGES72-BASENAME := genmodel/passages72.wsj02to21-gcg15-decoupled-fg-3sm-bd-x+efabp-+c_+b5000.syn.5-kenlm.bysent
PASSAGES72-LMEDEFAULT := 
PASSAGES96-BASENAME := genmodel/passages96.wsj02to21-gcg15-decoupled-fg-3sm-bd-x+efabp-+c_+b5000.syn.5-kenlm.bysent
PASSAGES96-LMEDEFAULT := 

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

define PASSAGES-FORM = 
scripts/$(1)%.lmeform: scripts/%.template.lmeform
	mkdir -p scripts
	cat $$< | sed 's/PASSAGESROI/$(1)/g;' > $$@
endef

define PASSAGES-FORMS 
$(foreach ROI, $(PASSAGES-ROI), $(eval $(call PASSAGES-FORM,$(ROI))))
endef

$(eval $(PASSAGES-FORMS))

genmodel/%.xlsx: $$(DEP-PASSAGES) $(PASSAGESDIR)/$$*.xlsx $(CONFIGDIR)/user-passages-directory.txt | genmodel
	cp $(word 1, $^) $@

%passages72.lineitems: genmodel/passages_eval.fromxlsx.csv | genmodel
	head -n+73 $(word 1, $^) | tail -n+2 | \
	awk '{for (i=3; i<7; i++) if ($$i != "") print $$i}' FPAT="([^ ]*)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' > $@

genmodel/passages72.linetoks: $$(DEP-TOKENIZER) genmodel/passages72.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $(word 1, $^) | $(word 2, $^) > $@

%passages96.lineitems: genmodel/passages_96concepts.fromxlsx.csv | genmodel
	tail -n+2 $(word 1, $^) | awk '{print $$3}' FPAT="([^ ]+)|(\"[^\"]+\")" | sed 's/^\"\|\"$$//g' > $@

genmodel/passages96.linetoks: $$(DEP-TOKENIZER) genmodel/passages96.lineitems $(TOKENIZERDIR)/ptb_tokenizer.sed $(CONFIGDIR)/user-tokenizer-directory.txt | genmodel
	cat $(word 1, $^) | $(word 2, $^) > $@

passages%.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-PASSAGES)/srcmodel/passages$$*.stripped.linetrees passages%.linetoks
	python $^ > $@

passages%.evmeasures: $$(DEP-PASSAGES) $(PASSAGESDIR)/passages_data_n$$*.csv \
$(RT-SCRIPTS)/rename_cols.py \
$(PASSAGES-SCRIPTS)/get_sampleids.py \
$(CONFIGDIR)/user-passages-directory.txt | genmodel
	cat $(word 1, $^) | sed 's/,/ /g;s/_sent/ /g;s/passage//g;s/con_name/docid sentid/g' | \
	python $(word 2, $^) subj_name subject con_name docid psc_LPostTemp boldLPostTemp psc_LAntTemp boldLAntTemp psc_LAngG boldLAngG psc_LIFG boldLIFG psc_LMFG boldLMFG psc_LIFGorb boldLIFGorb | \
	awk 'NR == 1 {print;}; NR > 1 {$$3 = $$3 - 1; print;};' | \
	python $(word 3, $^) > $@

passages%-batch: $(foreach roi, $(PASSAGES-ROI), $(foreach pred, $(PASSAGES-PREDS), $$(PASSAGES$$*-BASENAME).$(roi)passagesBasic.$(pred).$$(PASSAGES$$*-LMEDEFAULT)..lrt));

passages%.corr.txt: passages%.src.evmeasures $(PASSAGES-SCRIPTS)/getCorrBySubj.r
	cat $(word 1, $^) | $(word 2, $^) > $@
