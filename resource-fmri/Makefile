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
#  This directory contains reusable recipes for setting up reading time studies.
#  
################################################################################

################################################################################
#
#  Includes
#
#  $(RESOURCE-RT)/Makefile
#
#  Plus include statements to all makefiles necessary to build experiment-specific
#  prerequisites to these recipes.
#  
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
SECONDEXPANSION:

VPATH := $(RESOURCE-FMRI):$(VPATH):$(RESOURCE-GENERAL)

ifndef SAMPLE-START
SAMPLE-START := 0
endif

ROI := LPostTemp RPostTemp LPostTempZ RPostTempZ LAntTemp RAntTemp LAntTempZ RAntTempZ LIFG RIFG LIFGZ RIFGZ LIFGorb RIFGorb LIFGorbZ RIFGorbZ LMFG RMFG LMFGZ RMFGZ LAngG RAngG LAngGZ RAngGZ LHip RHip Hip LHipZ RHipZ HipZ LATL LIPL LPreM LPTL RATL AllROI 

FMRI-PRED-CONTROLS := sentpos wlen rate soundPower unigram
FMRI-PRED-BASES := wlen $(SURP) $(DLT) $(FJ) $(SEMDIST)
BYSENT-PREDS := nwrds $(foreach pred,$(FMRI-PRED-BASES),$(foreach agg,Mean Sum Max,$(pred)$(agg)))
BYSENT-PREDS += semCosDistMin semCosDistMax semCosDistMean semCosDistSum semDistMin semDistMax semDistMean semDistSum

define FMRI-FORM =
scripts/$(1)%.lmeform: scripts/%.template.lmeform
	mkdir -p scripts
	cat $$< | sed 's/FMRI-ROI/$(1)/g;' > $$@
endef

define FMRI-FORMS
$(foreach ROI, $(ROI), $(eval $(call FMRI-FORM,$(ROI))))
endef

$(eval $(FMRI-FORMS))

# ARGS:
#   1 = corpus name
#   2 = basename string
#   3 = ROI list
#   4 = LME formula name
#   5 = default LME args
define FMRI-HRF-EXPERIMENTS
$(foreach var, $(FMRI-PRED-BASES),$(1)-$(var)-$(4): $(foreach roi,$(3),$(2).$(roi)$(4).$(var).$(5)..lrt)
)
$(foreach var, $(FMRI-PRED-BASES),$(1)-$(var)-$(4)_%: $(foreach roi,$(3),$(2).$(roi)$(4).$(var).%..lrt);
)
$(1)-HRF-$(4): $(foreach var, $(FMRI-PRED-BASES), $(1)-$(var)-$(4))
$(1)-HRF-$(4)_%: $(foreach var, $(FMRI-PRED-BASES), $(1)-$(var)-$(4)_%);
endef

# ARGS:
#   1 = corpus name
#   2 = basename string
#   3 = ROI list
#   4 = LME formula name
#   5 = default LME args
define FMRI-HRF-CONTROLS
$(foreach var, $(FMRI-PRED-CONTROLS),$(1)-$(var)-$(4): $(foreach roi,$(3),$(2).$(roi)$(4).$(var).$(5)..lrt)
)
$(foreach var, $(FMRI-PRED-CONTROLS),$(1)-$(var)-$(4)_%: $(foreach roi,$(3),$(2).$(roi)$(4).$(var).%..lrt);
)
$(1)-HRFCONTROLS-$(4): $(foreach var, $(FMRI-PRED-CONTROLS), $(1)-$(var)-$(4))
$(1)-HRFCONTROLS-$(4)_%: $(foreach var, $(FMRI-PRED-CONTROLS), $(1)-$(var)-$(4)_%);
endef

# ARGS:
#   1 = corpus name
#   2 = basename string
#   3 = ROI list
#   4 = LME formula name
#   5 = default LME args
define FMRI-BYSENT-EXPERIMENTS
$(foreach var, $(BYSENT-PREDS),$(1)-$(var)-$(4): $(foreach roi,$(3),$(2).$(roi)$(4).$(var).$(5)..lrt)
)
$(foreach var, $(BYSENT-PREDS),$(1)-$(var)-$(4)_%: $(foreach roi,$(3),$(2).$(roi)$(4).$(var).%..lrt);
)
$(1)-BYSENT-$(4): $(foreach var, $(BYSENT-PREDS), $(1)-$(var)-$(4))
$(1)-BYSENT-$(4)_%: $(foreach var, $(BYSENT-PREDS), $(1)-$(var)-$(4)_%);
endef

################################################################################
#
#  Combined toks file creation
#
################################################################################

define KernelBlock
fMRIColumns:
  blocktitle: fMRI Column File
  paramtype: Text
  paramval:
  - text: Basename of file containing columns to convolve for fMRI analysis
    value: ''
fMRICorpus:
  blocktitle: fMRI Corpus
  instance_of: Corpus
  paramtype: Dropdown
endef
define CompositeBlock
fMRIDataParams:
  blockseq:
  - blocktype: fMRICorpus
  - blocktype: ParseParams
    kwargs:
      innerdelim: '-'
      instancename: Surprisal Metrics
  - blocktype: NgramParams
    kwargs:
      innerdelim: '-'
  - blocktype: fMRIColumns
  - blocktype: String
    value: '.hrf'
  blocktitle: fMRI Data Parameters
endef
define TargetBlock
fMRICoreEvmeasures:
  blocktitle: fMRI Data
  targetsuffix: .core.evmeasures
  family: Event Data
  blockseq:
  - blocktype: fMRIDataParams
    kwargs:
      innerdelim: .
endef
.PRECIOUS: %.hrf.itemmeasures
%.hrf.itemmeasures: %.t.itemmeasures $(FMRI-SCRIPTS)/hrf_convolve_predictors.py
	python $(word 2, $^) $(word 1, $^) > $@

%.bysent.itemmeasures: %.itemmeasures $(FMRI-SCRIPTS)/reducesents.py $$(word 1, $$(subst ., , %)).glove.linemeasures
	cat $(word 1, $^) | python $(word 2, $^) | paste -d' ' - <(cut -d' ' -f 2- $(word 3, $^)) > $@

.PRECIOUS: %.hrf.core.evmeasures
%.hrf.core.evmeasures : %.hrf.itemmeasures $$(word 1, $$(subst ., , %)).evmeasures \
$(RT-SCRIPTS)/merge_tables.py
	cat $< | python $(word 3, $^) $< $(word 2, $^) sampleid > $@

.PRECIOUS: %.bysent.core.evmeasures
%.bysent.core.evmeasures : %.bysent.itemmeasures $$(word 1, $$(subst ., , %)).evmeasures \
$(RT-SCRIPTS)/merge_tables.py
	cat $< | python $(word 3, $^) $< $(word 2, $^) sentid > $@

%.corr.txt: %.evmeasures $(FMRI-SCRIPTS)/corrAnalysis.r
	cat $(word 1, $^) | $(word 2, $^) > $@

