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

# VPATH += $(RESOURCE-CDR)

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-cdr-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CDRDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CDRDIR))))
RESOURCE-NAME := Deconvolutional Time Series Regression (CDR)
define RESOURCE-DESCR =

NAME: Deconvolutional Time Series Regression (CDR)
MB POINTER FILE: config/user-cdr-directory.txt
AVAILABILITY: Public
DESCRIPTION: A toolkit for jointly learning effect estimates and
temporal convolution shapes from time series data. 
endef
CDR-ERROR := $(call DEPWARN, $(CDR-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-CDR = $(error $(CDR-ERROR))
endif
endif


%/user-cdr-directory.txt: | %   
	echo '/data/compling/cdr' > $@

PYTHONPATH= $(shell cat $(CONFIGDIR)/user-cdr-directory.txt)
export PYTHONPATH

################################################################################
#
#  Recipes 
#
################################################################################

scripts/%cdrform: $(CDR-SCRIPTS)/$$*cdrform $(CDR-SCRIPTS)/get_prereqs_from_baseform.py $$(shell cat $(CDR-SCRIPTS)/%cdrform | python $(CDR-SCRIPTS)/get_prereqs_from_baseform.py)  |  scripts
	cp $< $@

bin/regress-cdr: $$(DEP-CDR) $(CDR-SCRIPTS)/regress-cdr.sh
	cp $< $@

bin/predict-cdr-dependonallparts: $$(DEP-CDR) $(CDR-SCRIPTS)/predict-cdr-dependonallparts.sh
	cp $< $@

.PRECIOUS: %.t.core.evmeasures
%.t.core.evmeasures: %.core.evmeasures $$(CDR-SCRIPTS)/compute_timestamps_from_RT.py
	cat $(word 1, $^) | python $(word 2, $^) > $@

%.y.evmeasures: %.evmeasures $$(RT-SCRIPTS)/filter_cols.awk
	cat $(word 1, $^) | awk -f $(word 2, $^) -v cols=fdur:fdurFP:fdurGP:time:subject:docid:sentid:word:correct:startofsentence:endofsentence:startofline:endofline:startofscreen:endofscreen:startoffile:endoffile:wdelta > $@

## TO SPEC
%.filt-evmeasures: $$(DEP-CDR) \
$$(basename $$(basename %)).core-evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).cdrform \
$(CDR-SCRIPTS)/infer_cdr_predictors.py \
$(RT-SCRIPTS)/filter_cols.awk \
$(RT-SCRIPTS)/accumulateMetrics.py \
$(RT-SCRIPTS)/rm_unfix_items.py \
$(RT-SCRIPTS)/futureMetrics.py \
$(RT-SCRIPTS)/spilloverMetrics.py \
$(RT-SCRIPTS)/parsePredictors.py \
$(RT-SCRIPTS)/rm_na_items.py \
$(CONFIGDIR)/user-cdr-directory.txt
	$(eval CORECOLS := $(sort $(shell echo $(shell cat $(word 2, $^) | python3 $(word 3, $^) $(shell cat $(word 11, $^))) $(shell echo $(subst +, ,$(subst _, ,$(subst .,,$(suffix $*))))) | python $(word 9, $^))))
	$(eval COLS := $(sort $(shell echo $(shell cat $(word 2, $^) | python3 $(word 3, $^) $(shell cat $(word 11, $^))) $(shell echo $(subst +, ,$(subst _, ,$(subst .,,$(suffix $*))))))))
	@echo Extracting columns: $(CORECOLS)
	@echo Computing columns: $(COLS)
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(CORECOLS))) $< | \
	python $(word 5, $^) -c $(shell echo $(COLS) | python $(word 9, $^) -a) | \
	python $(word 6, $^) | \
	python $(word 7, $^) -I -c $(shell echo $(COLS) | python $(word 9, $^) -f) | \
	python $(word 8, $^) -I -C $(shell echo $(COLS) | python $(word 9, $^) -s) | \
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(COLS))) | \
	python $(word 10, $^) > $@

%.filt.evmeasures: $$(DEP-CDR) \
$$(basename $$(basename %)).core.evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).cdrform \
$(CDR-SCRIPTS)/infer_cdr_predictors.py \
$(RT-SCRIPTS)/filter_cols.awk \
$(RT-SCRIPTS)/accumulateMetrics.py \
$(RT-SCRIPTS)/rm_unfix_items.py \
$(RT-SCRIPTS)/futureMetrics.py \
$(RT-SCRIPTS)/spilloverMetrics.py \
$(RT-SCRIPTS)/parsePredictors.py \
$(RT-SCRIPTS)/rm_na_items.py \
$(CONFIGDIR)/user-cdr-directory.txt
	$(eval CORECOLS := $(sort $(shell echo $(shell cat $(word 2, $^) | python3 $(word 3, $^) $(shell cat $(word 11, $^))) $(shell echo $(subst +, ,$(subst _, ,$(subst .,,$(suffix $*))))) | python $(word 9, $^))))
	$(eval COLS := $(sort $(shell echo $(shell cat $(word 2, $^) | python3 $(word 3, $^) $(shell cat $(word 11, $^))) $(shell echo $(subst +, ,$(subst _, ,$(subst .,,$(suffix $*))))))))
	@echo Extracting columns: $(CORECOLS)
	@echo Computing columns: $(COLS)
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(CORECOLS))) $< | \
	python $(word 5, $^) -c $(shell echo $(COLS) | python $(word 9, $^) -a) | \
	python $(word 6, $^) | \
	python $(word 7, $^) -I -c $(shell echo $(COLS) | python $(word 9, $^) -f) | \
	python $(word 8, $^) -I -C $(shell echo $(COLS) | python $(word 9, $^) -s) | \
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(COLS))) | \
	python $(word 10, $^) > $@

## TO SPEC
.PRECIOUS: %_cdr_outdir/finished.txt
%_cdr_outdir/finished.txt: $$(DEP-CDR) \
%.filt-evmeasures \
$$(word 1,$$(subst ., ,%)).evmeasures \
scripts/$$(word 1, $$(subst _, ,$$(subst .,,$$(suffix $$*)))).cdrform \
$(CONFIGDIR)/user-cdr-directory.txt \
$(CDR-SCRIPTS)/generate_config.py \
$(CDR-SCRIPTS)/get_split_id_fields.py \
$(CDRDIR)/cdr/bin/partition.py \
$(CDRDIR)/cdr/bin/train.py \
$(CDRDIR)/cdr/cdr.py \
$(CDRDIR)/cdr/cdrbayes.py
	mkdir -p $(dir $@)
	python3 $(word 7, $^) $(word 2, $^) -m 4 -n 3 -f $(shell cat $(word 3, $^) | python $(word 6, $^))
	cat $(word 3, $^) | python $(word 5, $^) $(word 1, $^) $(word 2, $^) $(dir $@) $(wordlist 2, $(words $(subst _, ,$(subst .,,$(suffix $*)))), $(subst _, ,$(subst .,,$(suffix $*))))
	python3 $(word 8, $^) $(dir $@)/config.ini
	echo "" > $@

.PRECIOUS: %.cdr_outdir/finished.txt
%.cdr_outdir/finished.txt: $$(DEP-CDR) \
%.filt.evmeasures \
$$(word 1,$$(subst ., ,%)).evmeasures \
scripts/$$(subst .,,$$(suffix $$(basename $$*))).cdrform \
$(CONFIGDIR)/user-cdr-directory.txt \
$(CDR-SCRIPTS)/generate_config.py \
$(CDR-SCRIPTS)/get_split_id_fields.py \
$(CDRDIR)/cdr/bin/partition.py \
$(CDRDIR)/cdr/bin/train.py \
$(CDRDIR)/cdr/cdrbase.py \
$(CDRDIR)/cdr/cdrbayes.py
	mkdir -p $(dir $@)
	python3 $(word 7, $^) $(word 2, $^) -m 4 -n 3 -f $(shell cat $(word 3, $^) | python $(word 6, $^))
	cat $(word 3, $^) | python $(word 5, $^) $(word 1, $^) $(word 2, $^) $(dir $@) $(subst _, ,$(subst .,,$(suffix $*)))
	python3 $(word 8, $^) $(dir $@)/config.ini
	echo "" > $@

## TO SPEC
%_cdr: %_cdr_outdir/finished.txt;

## WS: WHY IS THIS IN HERE TWICE?
%.cdr: %.cdr_outdir/finished.txt;

# WS: THESE AREN'T REAL DEPENDENCIES, AND SO IF THESE DON'T EXIST AFTER CDR CRASH, THEY BREAK
%_prd.resmeasures:  %_predicted.errors 
	cp $(basename $*).fitmodel_outdir/CDR/preds_$(subst held,test,$(subst expl,dev,$(subst fit,train,$(subst expl+held,expl-held,$(subst .,,$(suffix $*)))))).txt $@

%_obs.resmeasures:  %_predicted.errors
	cp $(basename $*).fitmodel_outdir/CDR/obs_$(subst held,test,$(subst expl,dev,$(subst fit,train,$(subst expl+held,expl-held,$(subst .,,$(suffix $*)))))).txt $@


#%_ptsignif: $$(DEP-CDR) \
$$(basename $$(basename $$(basename %))).$$(subst .,,$$(suffix $$(basename $$*))).filt.evmeasures \
$$(word 1,$$(subst ., ,%)).evmeasures \
$$(basename $$(basename $$(basename %))).$$(subst .,,$$(suffix $$(basename $$*))).$$(subst .,,$$(suffix $$(basename $$(basename $$*)))) \
$(CDRDIR)/cdr/bin/predict.py \
$(CDRDIR)/cdr/bin/pt.py \
$(CDRDIR)/cdr/bin/signif_table.py
#	python3 $(word 4, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -p $(subst .,,$(suffix $*))
#	python3 $(word 5, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -a -M loglik -p $(subst .,,$(suffix $*))
#	python3 $(word 6, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -H -M pt -p $(subst .,,$(suffix $*)) > $@

%_2stepsignif: $$(DEP-CDR) \
$$(basename $$(basename $$(basename %))).$$(subst .,,$$(suffix $$(basename $$*))).filt.evmeasures \
$$(word 1,$$(subst ., ,%)).evmeasures \
$$(basename $$(basename $$(basename %))).$$(subst .,,$$(suffix $$(basename $$*))).$$(subst .,,$$(suffix $$(basename $$(basename $$*)))) \
$(CDRDIR)/cdr/bin/convolve.py \
$(CDRDIR)/cdr/bin/lmer.py \
$(CDRDIR)/cdr/bin/lrt.py \
$(CDRDIR)/cdr/bin/signif_table.py
	python3 $(word 4, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -p $(subst .,,$(suffix $*)) 
	python3 $(word 5, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -z -p $(subst .,,$(suffix $*))
	python3 $(word 6, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -p $(subst .,,$(suffix $*)) -a
	python3 $(word 7, $^) $(basename $(basename $(basename $*))).$(subst .,,$(suffix $(basename $*))).$(subst .,,$(suffix $(basename $(basename $*))))_outdir/config.ini -H -M 2step -p $(subst .,,$(suffix $*)) > $@


