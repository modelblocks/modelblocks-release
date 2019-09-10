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
#  Include statements to this file should be preceded by includes to any 
#  Makefiles needed to produce the data to evaluate: <data-name>.core.evmeasures. 
#  For reading time experiments, this is usually resource-rt/Makefile and all 
#  Makefiles used to generate and combine experiment-specific data.
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

################################################################################
#
#  LMER recipes
#
################################################################################

# Fits an lme model based on the following template:
#
#   <evmeasure-baseline>.<formula-key>.<eval-params>.rdata
# 
# Command line arguments that handle field names must use "+" as a field separator.
# Distinct command line arguments must be separated in the target name by "_".
# For example, the following eval params chunk
#
#    -NF_-a_fjprob+dlt
#
# will pass the parameters -NF -a fjprob+dlt to the evaluation, which will treat
# "fjprob" and "dlt" as separate data fields (in this case, fixed effects to ablate/remove).
#
# Formula specifications must be in the local scripts/ directory and named "<key>.lmeform".
# They should consist of 4 lines: the dependent variable on the 1st line, the fixed effects
# on the second line, by_subject random slopes on the 3rd line, and any other random effects
# on the 4th line. Common formulas for latency experiments are in resource-lmefit/scripts/.
# mem.lmeform is designed for eye-tracking data, and spr.lmeform is designed for SPR data.
# You can copy, rename, and locally edit these according to your needs, provided you specify
# the correct key in the target for this recipe.
#
# For information about available parameters
# to the evaluation, run "resource-rt/scripts/evmeasures2lmefit.r -h" from the repository
# root.
#
#.PRECIOUS: %.rdata ##Disabled for now because it fills the disk quickly. Re-enable for simulations

bin/regress-lmer: $(LMEFIT-SCRIPTS)/regress-lmer.sh $(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	cp $< $@

bin/predict-lmer: $(LMEFIT-SCRIPTS)/predict-lmer.sh $(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	cp $< $@

bin/signif-lrt: $(LMEFIT-SCRIPTS)/signif-lrt.sh $(LMEFIT-SCRIPTS)/run_lrt.py \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	cp $< $@

define ParamVal
Partition:
  value: ''
  text: Data Partition
  kernel: LMEArgs4LRT
  cascade: Partition
Dev:
  value: -d
  text: Exploratory data
  kernel: Partition
Test:
  value: -t
  text: Confirmatory data
  kernel: Partition
Entire:
  value: -e
  text: Entire dataset
  kernel: Partition
SplitCols:
  value: -s
  text: Columns ('+'-delimited) on which to partition data
  cascade: SplitCols
  kernel: LMEArgs4LRT
FilterLines:
  value: -N
  text: Filter line starts/ends
  kernel: LMEArgs4LRT
FilterSents:
  value: -S
  text: Filter sentence starts/ends
  kernel: LMEArgs4LRT
FilterScreens:
  value: -C
  text: Filter screen starts/ends
  kernel: LMEArgs4LRT
FilterFiles:
  value: -F
  text: Filter file starts/ends
  kernel: LMEArgs4LRT
FilterPunc:
  value: -p
  text: Filter out phrase punctuation
  kernel: LMEArgs4LRT
LogDepVar:
  value: -l
  text: Log-transform the dependent variable
  kernel: LMEArgs4LRT
LogMain:
  value: -L
  text: Log-transform the main effect
  kernel: LMEArgs4LRT
BoxCox:
  value: -X
  text: Find and apply the best Box & Cox (1964) transformation to dependent variable
  kernel: LMEArgs4LRT
Interact:
  value: -I
  text: Do not include interaction between random slopes and intercepts
  kernel: LMEArgs4LRT
AddEffects:
  value: -A
  text: Names of effects to add ('+'-delimited)
  kernel: LMEArgs
  cascade: AddEffects
AblateEffects:
  value: -a
  text: Names of effects to ablate ('+'-delimited)
  kernel: LMEArgs
  cascade: AblateEffects
ExtraEffects:
  value: -x
  text: Names of any extra (non-main) effects to add ('+'-delimited)
  kernel: LMEArgs4LRT
  cascade: ExtraEffects
RestrictDomain:
  value: -R
  text: Restrict data to some subset
  kernel: LMEArgs4LRT
  cascade: RestrictDomain
endef
define KernelBlockDefs :=
BaselineFormula:
  blocktitle: LME Baseline Formula
  paramtype: Text
  paramval:
  - text: Basename of LME formula file
    value: ''
LMEArgs:
  blocktitle: LME command-line arguments
  paramtype: Dropdown
  nargs: '*'
LMEArgs4LRT:
  blocktitle: LME command-line arguments
  paramtype: Dropdown
  nargs: '*'
  instance_of: LMEArgs
Partition:
  blocktitle: Partition
  paramtype: Dropdown
SplitCols:
  blocktitle: Split columns
  paramtype: Text
  paramval:
  - text: Columns ('+'-delimited) on which to partition data
    value: ''
AddEffects:
  blocktitle: Add Effects 
  paramtype: Text
  paramval:
  - text: Names of effects to add ('+'-delimited)
    value: ''
AblateEffects:
  blocktitle: Ablate Effects 
  paramtype: Text
  paramval:
  - text: Names of effects to ablate ('+'-delimited)
    value: ''
ExtraEffects:
  blocktitle: Extra Effects 
  paramtype: Text
  paramval:
  - text: Names of any extra (non-main) effects to add ('+'-delimited)
    value: ''
RestrictDomain:
  blocktitle: Restrict Domain 
  paramtype: Text
  paramval:
  - text: Basename of file containing data filtering instructions
    value: ''
endef
define TargetBlockDefs :=
LMEFitfMRI:
  blocktitle: fMRI LME Fit
  targetsuffix: .lme
  family: Linear Mixed-Effects (LME)
  blockseq:
  - blocktype: fMRIDataParams
    kwargs:
      innerdelim: .
  - blocktype: String
    value: .
  - blocktype: BaselineFormula
  - blocktype: String
    value: .
  - blocktype: LMEArgs
LMEFitRT:
  blocktitle: Reading Time LME Fit
  targetsuffix: .lme
  family: Linear Mixed-Effects (LME)
  blockseq:
  - blocktype: RTDataParams
    kwargs:
      innerdelim: .
  - blocktype: String
    value: .
  - blocktype: BaselineFormula
  - blocktype: String
    value: .
  - blocktype: LMEArgs
    kwargs:
      outerdelim: _
BMEFitfMRI:
  blocktitle: fMRI BME Fit
  targetsuffix: .bme
  family: Bayesian Mixed-Effects (BME)
  blockseq:
  - blocktype: fMRIDataParams
    kwargs:
      innerdelim: .
  - blocktype: String
    value: .
  - blocktype: BaselineFormula
  - blocktype: String
    value: .
  - blocktype: LMEArgs
BMEFitRT:
  blocktitle: Reading Time BME Fit
  targetsuffix: .bme
  family: Bayesian Mixed-Effects (BME)
  blockseq:
  - blocktype: RTDataParams
    kwargs:
      innerdelim: .
  - blocktype: String
    value: .
  - blocktype: BaselineFormula
  - blocktype: String
    value: .
  - blocktype: LMEArgs
    kwargs:
      outerdelim: _
endef
# .PRECIOUS: %.tokmeasures
# %.tokmeasures: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r \
# genmodel/$$(basename $$(basename $$*)).$$(subst .,,$$(suffix $$(basename $$*)))_$$(shell python $$(LMEFIT-SCRIPTS)/getEffectsFromCLI.py \
# $$(subst .,,$$(suffix $$*))).filt.evmeasures \
# scripts/$$(subst .,,$$(suffix $$(basename $$*))).lmeform \
# $(LMEFIT-SCRIPTS)/lmetools.r \
# $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
# $(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
# 	$< $(word 2, $^) $@ $(subst _, ,$(subst .,,$(suffix $*))) -b $(word 3, $^) -T > $*.lme

.PRECIOUS: %.lmefit %.lme %.lme.rdata
%.lme.rdata %.lmefit %.lme: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r \
genmodel/$$(basename $$(basename $$*)).$$(subst .,,$$(suffix $$(basename $$*)))_$$(shell python $$(LMEFIT-SCRIPTS)/getEffectsFromCLI.py \
$$(subst .,,$$(suffix $$*))).filt.evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).lmerform \
$(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$< $(word 2, $^) $*.lme.rdata $(subst _, ,$(subst .,,$(suffix $*))) -b $(word 3, $^) > $*.lme

.PRECIOUS: %.bmefit %.bme %.bme.rdata
%.bme.rdata %.bmefit %.bme: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r \
genmodel/$$(basename $$(basename $$*)).$$(subst .,,$$(suffix $$(basename $$*)))_$$(shell python $$(LMEFIT-SCRIPTS)/getEffectsFromCLI.py \
$$(subst .,,$$(suffix $$*))).filt.evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).lmerform \
$(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$< $(word 2, $^) $*.bme.rdata $(subst _, ,$(subst .,,$(suffix $*))) -b $(word 3, $^) -m bme > $*.bme

.PRECIOUS: %.lmfit %.lm %.lm.rdata
%.lm.rdata %.lmfit %.lm: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r \
genmodel/$$(basename $$(basename $$*)).$$(subst .,,$$(suffix $$(basename $$*)))_$$(shell python $$(LMEFIT-SCRIPTS)/getEffectsFromCLI.py \
$$(subst .,,$$(suffix $$*))).filt.evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).lmerform \
$(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$< $(word 2, $^) $*.lm.rdata $(subst _, ,$(subst .,,$(suffix $*))) -b $(word 3, $^) -m lm > $*.lm

sp :=
sp +=
pc := %

## TO SPEC:
.PRECIOUS: %_lmer %_lmer.rdata
%_lmer %_lmer.rdata: $$(DEP-RHACKS)  $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r \
                     genmodel/$$(basename $$*).$$(subst $$(sp),_,$$(filter-out -$$(pc),$$(subst _, ,$$(subst .,,$$(suffix $$*)))))_$$(shell python $$(LMEFIT-SCRIPTS)/getEffectsFromCLI.py \
                     $$(subst -a,-a_,$$(subst -A,-A_,$$(subst .,,$$(suffix $$*))))).filt-evmeasures \
                     scripts/$$(filter-out -$$(pc),$$(subst _, ,$$(subst .,,$$(suffix $$*)))).lmerform \
                     $(LMEFIT-SCRIPTS)/lmetools.r \
                     $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
                     $(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$< $(word 2, $^) $*_lmer.rdata $(subst -A,-A ,$(subst -a,-a ,$(filter -%,$(subst _, ,$(subst .,,$(suffix $*)))))) -b $(word 3, $^) > $*_lmer


# Runs an LRT evaluation based on the following template:
#
#    <path>/<evmeasures-basename>.<formula-key>.<main-effect>.<eval-params>.<anova-params>.diamond.lrt
#
# Formula specifications must be in the local scripts/ directory and named "<key>.lmeform".
# They should consist of 4 lines: the dependent variable on the 1st line, the fixed effects
# on the second line, by_subject random slopes on the 3rd line, and any other random effects
# on the 4th line. Common formulas for latency experiments are in resource-lmefit/scripts/.
# mem.lmeform is designed for eye-tracking data, and spr.lmeform is designed for SPR data.
# You can copy, rename, and locally edit these according to your needs, provided you specify
# the correct key in the target for this recipe.
#
# For information about available parameters
# to the evaluation, run "resource-rt/scripts/evmeasures2lmefit-mem.r -h" from the repository
# root.
#
# Example:
#
#    genmodel/ucl.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.kenlm.fp.mem.fjprob.-NSCFf..lrt
#
#
define ParamVal
LRTNull:
  text: NONE CURRENTLY AVAILABLE
  value: ''
  kernel: LRTArgs
endef
define KernelBlock
LMEPredictor:
  blocktitle: LME Predictor
  paramtype: Text
  paramval:
  - text: 'Name of LME predictor column'
    value: ''
LRTArgs:
  blocktitle: LRT command-line args
  paramtype: Dropdown
  nargs: '*'
endef
define TargetBlock
RTLRTSignif:
  blocktitle: Reading Time LRT Significance Test
  targetsuffix: .lrt
  family: Likelihood Ratio Test (LRT)
  blockseq:
  - blocktype: RTDataParams
    kwargs:
      innerdelim: '.'
  - blocktype: String
    value: '.'
  - blocktype: BaselineFormula
  - blocktype: String
    value: '.'
  - blocktype: LMEPredictor
  - blocktype: String
    value: '.'
  - blocktype: LMEArgs4LRT
  - blocktype: String
    value: '.'
  - blocktype: LRTArgs
fMRILRTSignif:
  blocktitle: fMRI LRT Significance Test
  targetsuffix: .lrt
  family: Likelihood Ratio Test (LRT)
  blockseq:
  - blocktype: fMRIDataParams
    kwargs:
      innerdelim: '.'
  - blocktype: String
    value: '.'
  - blocktype: BaselineFormula
  - blocktype: String
    value: '.'
  - blocktype: LMEPredictor
  - blocktype: String
    value: '.'
  - blocktype: LMEArgs4LRT
  - blocktype: String
    value: '.'
  - blocktype: LRTArgs
endef
.PRECIOUS: %.lrtsignif %.lrt
%.lrtsignif %.lrt: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/lmefit2lrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$(wordlist 1,3,$^) $(subst _, ,$(subst .,,$(suffix $*))) > $@

.PRECIOUS: %.permsignif %.perm
%.permsignif %.perm: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/lmefit2lrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$(wordlist 1,3,$^) $(subst _, ,$(subst .,,$(suffix $*))) -perm > $@

## TO SPEC:
.PRECIOUS: %_lrtsignif
%_lrtsignif: $$(DEP-RHACKS)  $(LMEFIT-SCRIPTS)/lmefit2lrtsignif.r \
             $$(basename $$(basename $$*)).-a$$(filter-out -$$(pc),$$(subst _, ,$$(subst .,,$$(suffix $$*))))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
             $$(basename $$(basename $$*)).-A$$(filter-out -$$(pc),$$(subst _, ,$$(subst .,,$$(suffix $$*))))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
             $(abspath $(RHACKS-SCRIPTS)/mer-utils.R)  $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
             $(LMEFIT-SCRIPTS)/lmetools.r \
             $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
             $(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$(wordlist 1,3,$^) $(filter -%,$(subst _, ,$(subst .,,$(suffix $*)))) > $@


# Runs a diamond LRT evaluation based on the following template:
#     
#    <path>/<evmeasures-basename>.<formula-key>.<effect1>_<effect2>.<eval-params>.<anova-params>.diamond.lrtsignif
#
# Formula specifications must be in the local scripts/ directory and named "<key>.lmeform".
# They should consist of 4 lines: the dependent variable on the 1st line, the fixed effects
# on the second line, by_subject random slopes on the 3rd line, and any other random effects
# on the 4th line. Common formulas for latency experiments are in resource-lmefit/scripts/.
# mem.lmeform is designed for eye-tracking data, and spr.lmeform is designed for SPR data.
# You can copy, rename, and locally edit these according to your needs, provided you specify
# the correct key in the target for this recipe.
#
# For information about available parameters
# to the evaluation, run "resource-rt/scripts/evmeasures2lmefit-mem.r -h" from the repository
# root.
#    
# Example:
#
#    genmodel/ucl.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.kenlm.fp.mem.fjprob_dlt.-NSCFf..diamond.lrtsignif 
#    
define CompositeBlock
LMEPredictors:
  blocktitle: Two LME predictors (for diamond LRT signif)
  blockseq:
  - blocktype: LMEPredictor
  - blocktype: LMEPredictor
endef
define TargetBlock
RTLRTDaimSignif:
  blocktitle: Diamond Reading Time LRT Significance Test
  targetsuffix: .dlrt
  family: Likelihood Ratio Test (LRT)
  blockseq:
  - blocktype: RTDataParams
    kwargs:
      innerdelim: '.'
  - blocktype: String
    value: '.'
  - blocktype: BaselineFormula
  - blocktype: String
    value: '.'
  - blocktype: LMEPredictors
    kwargs:
      innerdelim: '_'
  - blocktype: String
    value: '.'
  - blocktype: LMEArgs
  - blocktype: String
    value: '.'
  - blocktype: LRTArgs
fMRILRTDaimSignif:
  blocktitle: Diamond fMRI LRT Significance Test
  targetsuffix: .dlrt
  family: Likelihood Ratio Test (LRT)
  blockseq:
  - blocktype: fMRIDataParams
    kwargs:
      innerdelim: '.'
  - blocktype: String
    value: '.'
  - blocktype: BaselineFormula
  - blocktype: String
    value: '.'
  - blocktype: LMEPredictors
    kwargs:
      innerdelim: '_'
  - blocktype: String
    value: '.'
  - blocktype: LMEArgs
  - blocktype: String
    value: '.'
  - blocktype: LRTArgs
endef
.PRECIOUS: %.diamond.lrtsignif %.dlrt
%.diamond.lrtsignif %.dlrt: $$(DEP-RHACKS) $(LMEFIT-SCRIPTS)/lmefit2diamlrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_-a_$$(word 2,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$$(basename $$(basename $$(basename $$*))).-a_$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_-A_$$(word 2,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_$$(subst .,,$$(suffix $$(basename %))).lme.rdata \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$(LMEFIT-SCRIPTS)/lmetools.r
	$(wordlist 1,5,$^) -t '$(subst _,-vs-,$(subst .,,$(suffix $(basename $(basename $*)))))' \
  -l $(word 1, $(subst _, ,$(subst .,,$(suffix $(basename $(basename $*)))))) \
  -r $(word 2, $(subst _, ,$(subst .,,$(suffix $(basename $(basename $*)))))) > $@

## TO SPEC:
.PRECIOUS: %_dlrtsignif
%_dlrtsignif: $$(DEP-RHACKS)  $(LMEFIT-SCRIPTS)/lmefit2diamlrtsignif.r \
              $$(basename $$(basename $$*)).-a$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))+$$(word 2,$$(subst _, ,$$(suffix $$*)))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
              $$(basename $$(basename $$*)).-a$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))_-A$$(word 2,$$(subst _, ,$$(suffix $$*)))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
              $$(basename $$(basename $$*)).-A$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))_-a$$(word 2,$$(subst _, ,$$(suffix $$*)))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
              $$(basename $$(basename $$*)).-A$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))+$$(word 2,$$(subst _, ,$$(suffix $$*)))_$$(subst .,,$$(suffix $$(basename $$*))).rdata \
              $(abspath $(RHACKS-SCRIPTS)/mer-utils.R)  $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
              $(LMEFIT-SCRIPTS)/lmetools.r \
              $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
              $(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	$(wordlist 1,5,$^) -t '$(subst $(sp),-vs-,$(filter-out -%,$(subst _, ,$(subst .,,$(suffix $*)))))' \
		-l $(word 1,$(subst _, ,$(subst .,,$(suffix $*)))) \
		-r $(word 2,$(subst _, ,$(suffix $*))) > $@


# Generates an easy-to-read table of summary data
# from one or more %.lrtsignif files. Pulls in all %.lrtsignif files
# files in the specified directory.

.PRECIOUS: /%.signiftable.txt
/%.signiftable.txt: $(LMEFIT-SCRIPTS)/extract_signif.py
	python $^ $(wildcard /$(dir $*)*lrt) > $@

.PRECIOUS: %.signiftable.txt
%.signiftable.txt: $$(abspath $$@)
	$(info )

%.error.evmeasures: genmodel/$$(basename $$(basename $$(basename $$*))).core.evmeasures \
$(LMEFIT-SCRIPTS)/lmefit2error.r \
$$(basename $$(basename $$*)).-a_$$(subst .,,$$(suffix $$(basename $$*)))_-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$*))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix %)).lme.rdata \
$$(basename $$(basename $$*)).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$*))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix %)).lme.rdata \
$(abspath $(RHACKS-SCRIPTS)/mtoolbox.R) $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R)
	cat $< | $(word 2, $^) -b $(word 3, $^) -m $(word 4, $^) -o $@ $(subst .,,$(suffix $*))

.PRECIOUS: /%.error.multi.evmeasures
/%.error.multi.evmeasures: $$(sort $$(foreach file,$$(wildcard /$$(dir %)*.lrtsignif), $$(basename $$(basename $$(file)))$$(suffix $$*).error.evmeasures))
	paste -d' ' <(cut -d' ' -f-5 $<) $(foreach file,$^,<(cut -d' ' -f6-7 $(file))) > $@

/%.error.multi.tokmeasures: /%.error.multi.evmeasures
	cat $^ | python ../resource-rt/scripts/evmeasures2itemmeans.py -c fdur .*SqErrReduc > $@
