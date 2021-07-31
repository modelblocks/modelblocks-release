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
#  include $(RESOURCE-LTREES)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-REGRESSION) $(RESOURCE-LMEFIT) $(RESOURCE-DTSR)

## literal space character for string conversions
empty:=
space:= $(empty) $(empty)


################################################################################
#
#  Recipes
#
################################################################################

#$(REGRESSION-SCRIPTS)/signif-pt.sh: $(REGRESSION-SCRIPTS)/pt.py;

#$(REGRESSION-SCRIPTS)/signif-ct.sh: $(REGRESSION-SCRIPTS)/pt.py;

## build generic signif testing executables...
bin/signif-%: $(REGRESSION-SCRIPTS)/signif-%.sh  |  bin
	cp $< $@

## obsolete?
%-avg.itemmeasures: $$(basename %).all-itemmeasures $(REGRESSION-SCRIPTS)/average_by_group.py
	cat $< | python $(word 2, $^) $(subst .,,$(suffix $*)) > $@


################################################################################
#
#  Items to assemble predictors
#
################################################################################    

## apply predictor modifications: <data>.<predr>(_<predr>)*_<offset>_spilled.prdmeasures
%_spilled.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/spilloverMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(subst ., ,$(basename $(subst _,.,$(subst .,,$(suffix $*))))) -n $(subst .,,$(suffix $(subst _,.,$(suffix $*)))) >  $@
%_spilled.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)

## apply predictor modifications: <data>.<predr>_<offset>_spilled.prdmeasures for second spilled over predictor
%_altspilled.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/spilloverMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(word 1, $(subst _, ,$(subst .,,$(suffix $*)))) -n $(word 2, $(subst _, ,$(subst .,,$(suffix $*)))) >  $@
%_altspilled.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)

## apply predictor modifications: <data>.<predr>_<offset>_spilled.prdmeasures for third spilled over predictor
%_altaltspilled.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/spilloverMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(word 1, $(subst _, ,$(subst .,,$(suffix $*)))) -n $(word 2, $(subst _, ,$(subst .,,$(suffix $*)))) >  $@
%_altaltspilled.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)

## apply predictor modifications: <data>.<predr>_cumued.prdmeasures
%_cumued.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/accumulateMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
%_cumued.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)

## apply predictor modifications: <data>.<predr>_<offset>_futured.prdmeasures
%_futured.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/futureMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
%_futured.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)

## apply predictor modifications: <data>.<predr>_<predr>_delta.prdmeasures
%_delta.prdmeasures:  $$(basename %).prdmeasures  $(RT-SCRIPTS)/deltaMetrics.py
	cat $(word 1,$^)  |  python $(word 2,$^) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@
%_delta.resmeasures:  $$(basename %).resmeasures
	cp $(abspath $<) $(abspath $@)
#	ln -sf $(abspath $<) $(abspath $@)


################################################################################
#
#  Items to filter ('censor') responses (e.g. remove end-of-sentence responses)
#
################################################################################    

## apply censorship (e.g. remove end-of-sentence responses) (prdmeasures theoretically unaffected, but if it contains responses, needs censoring too)
%.prdmeasures:  $(REGRESSION-SCRIPTS)/$$(subst .,,$$(suffix $$*)).cens_params.ini  $$(basename %).prdmeasures  $(REGRESSION-SCRIPTS)/censor.py
#	cat $(word 2,$^)  |  python3 $(word 3,$^) -c $(word 1,$^)  >  $@
	cp $(abspath $(word 2,$^)) $(abspath $@)
#	ln -sf $(abspath $(word 2,$^)) $(abspath $@)
%.resmeasures:  $(REGRESSION-SCRIPTS)/$$(subst .,,$$(suffix $$*)).cens_params.ini  $$(basename %).resmeasures  $(REGRESSION-SCRIPTS)/censor.py
	cat $(word 2,$^)  |  python3 $(word 3,$^) -c $(word 1,$^)  >  $@

# ## obsolete, related to eye scan-path...
# %.coarse.prdmeasures: $$(basename %).prdmeasures $(REGRESSION-SCRIPTS)/removeNanFdurs.py
# 	cat $(word 1,$^) | python3 $(word 2,$^) > $@
# %.coarse.resmeasures: $$(basename %).resmeasures 
# 	cp $^ $@


################################################################################
#
#  Items to partition responses
#
################################################################################    

## apply partitioning (prdmeasures unaffected)
%_part.prdmeasures:  genmodel/$$(notdir $$(word 1, $$(subst ., ,$$*))).part_params.ini  $$(basename %).prdmeasures  $(REGRESSION-SCRIPTS)/partition.py  |  genmodel
	cp $(abspath $(word 2,$^)) $(abspath $@)
#	ln -sf $(abspath $(word 2,$^)) $(abspath $@)
%_part.resmeasures:  genmodel/$$(notdir $$(word 1, $$(subst ., ,$$*))).part_params.ini  $$(basename %).resmeasures  $(REGRESSION-SCRIPTS)/partition.py  |  genmodel
	cat $(word 2,$^)  |  python3 $(word 3, $^) -c $(word 1, $^) -p $(subst +, ,$(subst .,,$(suffix $*)))  >  $@


################################################################################
#
#  Items to regress, producing a fit (for prediction)
#
################################################################################    

## generic fit files: <data>.<fit-part>.<baseline>_(<predr>_)*<regrmodel>.fitmodel
.PRECIOUS: %.fitmodel %.coeffs
%.fitmodel %.coeffs:  bin/regress-$$(subst .,,$$(suffix $$(subst _,.,$$*)))  \
                      genmodel/$$(notdir $$(basename $$*)).$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))_part.prdmeasures  \
                      genmodel/$$(notdir $$(basename $$*)).$$(word 1,$$(subst _, ,$$(subst .,,$$(suffix $$*))))_part.resmeasures  \
                      scripts/$$(subst .,,$$(word 2,$$(subst _, ,$$(suffix $$*))))$$(suffix $$(subst _,.,$$*))form  |  bin  genmodel
	$^ $(wordlist 3,100,$(subst ., ,$(basename $(subst _,.,$(suffix $*)))))  >  $@  2>  $@.log


################################################################################
#
#  Error files from prediction (for signif testing)
#
################################################################################    

## generic predicted errors file: <data>.<fit-part>.<baseline>_(<predr>_)*<regrmodel>.<eval-part>_predicted.errors
.PRECIOUS: %_predicted.errors
%_predicted.errors %_prd.resmeasures %_obs.resmeasures:  bin/predict-$$(subst .,,$$(suffix $$(subst _,.,$$(basename $$*))))  \
                                                         genmodel/$$(notdir $$(basename $$*)).fitmodel  \
                                                         genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.prdmeasures \
                                                         genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.resmeasures  |  bin  genmodel
	$^  >  $*_predicted.errors  2>  $*_predicted.errors.log

#.PRECIOUS: %_predicted.errors
##%_predicted.errors:  bin/predict-$$(subst .,,$$(suffix $$(subst _,.,$$(basename $$*))))-dependonallparts  \
                     genmodel/$$(notdir $$(basename $$*)).fitmodel  \
                     genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.prdmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.resmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).fit_part.prdmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).expl_part.prdmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).held_part.prdmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).fit_part.resmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).expl_part.resmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*))).held_part.resmeasures  |  bin  genmodel
#%_predicted.errors:  bin/predict-$$(subst .,,$$(suffix $$(subst _,.,$$(basename $$*))))-dependonallparts  \
                     genmodel/$$(notdir $$(basename $$*)).fitmodel  \
                     genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.prdmeasures \
                     genmodel/$$(notdir $$(basename $$(basename $$*)))$$(suffix $$*)_part.resmeasures  |  bin  genmodel
#	$^  >  $@  2>  $@.log


# ## generic errors file: <data>.<baseline>_(<predr>_)*<regrmodel>.errors
# %.errors:  bin/calc-errors  %_predicted.resmeasures  $$(basename $$(basename %)).$$(word 2,$$(subst +, ,$$(suffix $$(basename $$*))))-part.resmeasures
# 	$^  >  $@


################################################################################
#
#  Items for significance testing
#
################################################################################    

## generic signif file: <data>.<fit-part>_<baseline>_(<predr>_)*<regrmodel>.<eval-part>.(<predr>_)*<signif-type>signif
.PRECIOUS: %signif
%signif:  bin/signif-$$(lastword $$(subst _, ,$$(suffix $$*)))  \
	    $$(foreach abl,$$(shell python $(REGRESSION-SCRIPTS)/infer_ablation_powerset.py  $$(subst ., ,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))), genmodel/$$(notdir $$(basename $$(basename $$(basename $$*)))).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))_$$(abl)_$$(lastword $$(subst _, ,$$(basename $$(basename $$*))))$$(suffix $$(basename $$*))_predicted.errors)  |  bin  genmodel
	$^  >  $@  2>  $@.log

## WS: NON-GENERIC DEPENDENCIES
.PRECIOUS: %ctsignif
%ctsignif:  bin/signif-ct-nongeneric  \
	    $$(foreach abl,$$(shell python $(REGRESSION-SCRIPTS)/infer_ablation_powerset.py  $$(subst ., ,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))), genmodel/$$(notdir $$(basename $$(basename $$(basename $$*)))).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))_$$(abl)_$$(lastword $$(subst _, ,$$(basename $$(basename $$*))))$$(suffix $$(basename $$*))_prd.resmeasures) \
	    $$(foreach abl,$$(shell python $(REGRESSION-SCRIPTS)/infer_ablation_powerset.py  $$(subst ., ,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$*)))))), genmodel/$$(notdir $$(basename $$(basename $$(basename $$*)))).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))_$$(abl)_$$(lastword $$(subst _, ,$$(basename $$(basename $$*))))$$(suffix $$(basename $$*))_obs.resmeasures) \
	    |  bin  genmodel
	$^  >  $@  2>  $@.log


