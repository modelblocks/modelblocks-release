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
#  Makefiles needed to produce the data to evaluate: <data-name>.full.evmeasures. 
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
# to the evaluation, run "resource-rt/scripts/evmeasures2lmefit-mem.r -h" from the repository
# root.
#
.PRECIOUS: %.rdata
%.rdata:  $(LMEFIT-SCRIPTS)/evmeasures2lmefit.r genmodel/$$(basename $$(basename $$*)).full.evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).lmeform 
	$< $(word 2, $^) $@ $(subst _, ,$(subst .,,$(suffix $*))) -b $(word 3, $^) > $*.lmefit

# Runs a diamond LME evaluation based on the following template:
#
#    <path>/<evmeasures-basename>.<formula-key>.<main-effect>.<eval-params>.<anova-params>.diamond.lrtsignif
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
#    genmodel/amateur.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.kenlm.fp.mem.fjprob.-NSCFf..lrtsignif
#
#
.PRECIOUS: %.lrtsignif
%.lrtsignif:  $(LMEFIT-SCRIPTS)/lmefit2lrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst .,,$$(suffix $$(basename $$(basename $$*))))_-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$(RHACKS-SCRIPTS)/mer-utils.R $(RHACKS-SCRIPTS)/regression-utils.R
	$(words 2,3,$^) $(subst _, ,$(subst .,,$(suffix $*))) > $@

# Same as above, only don't add main effect as a random slope
.PRECIOUS: %.noadd.lrtsignif
%.noadd.lrtsignif:  $(LMEFIT-SCRIPTS)/lmefit2lrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst .,,$$(suffix $$(basename $$(basename $$*))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$$(basename $$(basename $$(basename $$*))).-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$(RHACKS-SCRIPTS)/mer-utils.R $(RHACKS-SCRIPTS)/regression-utils.R
	$(words 2,3,$^) $(subst _, ,$(subst .,,$(suffix $*))) > $@

  
# Runs a diamond LME evaluation based on the following template:
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
#    genmodel/amateur.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.kenlm.fp.mem.fjprob_dlt.-NSCFf..diamond.lrtsignif 
#    
.PRECIOUS: %.diamond.lrtsignif
%.diamond.lrtsignif: $(LMEFIT-SCRIPTS)/lmefit2diamlrtsignif.r \
$$(basename $$(basename $$(basename $$*))).-a_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$$(basename $$(basename $$(basename $$*))).-a_$$(word 2,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_-A_$$(word 2,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))+$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$$(basename $$(basename $$(basename $$*))).-a_$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$(basename $$(basename $$*))))))_-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$$(basename $$(basename $$(basename $$*))).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix $$(basename %))).rdata \
$(RHACKS-SCRIPTS)/mer-utils.R $(RHACKS-SCRIPTS)/regression-utils.R
	$(words 2,5) -t '$(subst _,-vs-,$(subst .,,$(suffix $(basename $(basename $*)))))' \
  -l $(word 1, $(subst _, ,$(subst .,,$(suffix $(basename $(basename $*)))))) \
  -r $(word 2, $(subst _, ,$(subst .,,$(suffix $(basename $(basename $*)))))) > $@

# Generates an easy-to-read table of summary data
# from one or more %.lrtsignif files. Pulls in all %.lrtsignif files
# files in the specified directory.

.PRECIOUS: /%.signiftable.txt
/%.signiftable.txt: $(LMEFIT-SCRIPTS)/extract_signif.py
	python $^ $(wildcard /$(dir $*)*.lrtsignif) > $@

.PRECIOUS: %.signiftable.txt
%.signiftable.txt: $$(abspath $$@)
	$(info )

%.error.evmeasures: genmodel/$$(basename $$(basename $$(basename $$*))).full.evmeasures \
$(LMEFIT-SCRIPTS)/lmefit2error.r \
$$(basename $$(basename $$*)).-a_$$(subst .,,$$(suffix $$(basename $$*)))_-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$*))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix %)).rdata \
$$(basename $$(basename $$*)).-A_$$(subst _,+,$$(subst .,,$$(suffix $$(basename $$*))))_-c_$$(word 1, $$(subst ., ,$$*))_$$(subst .,,$$(suffix %)).rdata \
$(RHACKS-SCRIPTS)/mtoolbox.R $(RHACKS-SCRIPTS)/mer-utils.R $(RHACKS-SCRIPTS)/regression-utils.R
	cat $< | $(word 2, $^) -b $(word 3, $^) -m $(word 4, $^) -o $@ $(subst .,,$(suffix $*))

.PRECIOUS: /%.error.multi.evmeasures
/%.error.multi.evmeasures: $$(sort $$(foreach file,$$(wildcard /$$(dir %)*.lrtsignif), $$(basename $$(basename $$(file)))$$(suffix $$*).error.evmeasures))
	paste -d' ' <(cut -d' ' -f-5 $<) $(foreach file,$^,<(cut -d' ' -f6-7 $(file))) > $@

/%.error.multi.tokmeasures: /%.error.multi.evmeasures
	cat $^ | python ../resource-rt/scripts/evmeasures2itemmeans.py -c fdur .*SqErrReduc > $@
