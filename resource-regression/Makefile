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

VPATH += $(RESOURCE-REGRESSION)

################################################################################
#
#  Recipes
#
################################################################################

empty:=
space:= $(empty) $(empty)

## apply consorship (e.g. remove end-of-sentence responses) (prdmeasures unaffected)
%.prdmeasures:  $(REGRESSION-SCRIPTS)/$$(subst .,,$$(suffix $$*)).cens_params.ini  $$(basename %).prdmeasures  $(REGRESSION-SCRIPTS)/censor.py
	ln -s $(word 2,$^) $@
%.resmeasures:  $(REGRESSION-SCRIPTS)/$$(subst .,,$$(suffix $$*)).cens_params.ini  $$(basename %).resmeasures  $(REGRESSION-SCRIPTS)/censor.py
	cat $(word 2,$^)  |  python $(word 3,$^) $(word 1,$^)  >  $@


## apply partitioning (prdmeasures unaffected)
%-part.prdmeasures:  genmodel/$$(word 1, $$(subst ., ,$$*)).part_params.ini  $$(basename %).prdmeasures  $(REGRESSION-SCRIPTS)/partition.py 
	ln -s $(word 2,$^) $@
%-part.resmeasures:  genmodel/$$(word 1, $$(subst ., ,$$*)).part_params.ini  $$(basename %).resmeasures  $(REGRESSION-SCRIPTS)/partition.py
	cat $(word 2,$^)  |  python $(word 3, $^) -c $(word 1, $^) -p $(subst +, ,$(subst .,,$(suffix $*)))  >  $@


## generic fit files: <data>.<baseline>_(<predr>_)*<regrmodel>.fitmodel
%.fitmodel %.coeffs:  bin/regressor-$$(subst .,,$$(suffix $$(subst _,.,$$*)))  \
                      $$(basename %)-part.prdmeasures  \
                      $$(basename %)-part.resmeasures  \
                      bin/$$(subst .,,$$(word 1,$$(subst _, ,$$(suffix $$*))))$$(suffix $$(subst _,.,$$*))form
	$^ $(wordlist 2,100,$(subst ., ,$(basename $(subst _,.,$(suffix $*)))))


## generaic predicted errors file: <data>.<fit-part>_<baseline>_(<predr>_)*<regrmodel>.<eval-part>_predicted.errors
%_predicted.errors:  bin/predictor-$$(subst .,,$$(suffix $$(subst _,.,$$(basename $$*))))  \
                     $$(basename $$(basename %))$$(word 1,$$(subst _, ,$$(suffix $$(basename $$*)))).$$(subst $$(space),_,$$(wordlist 2,100,$$(subst _, ,$$(suffix $$(basename $$*))))).fitmodel  \
           	     $$(basename $$(basename %))$$(suffix $$*)-part.prdmeasures
	$^  >  $@


# ## generic errors file: <data>.<baseline>_(<predr>_)*<regrmodel>.errors
# %.errors:  bin/calc-errors  %_predicted.resmeasures  $$(basename $$(basename %)).$$(word 2,$$(subst +, ,$$(suffix $$(basename $$*))))-part.resmeasures
# 	$^  >  $@


## generic signif file: <data>.<fit-part>_<baseline>_(<predr>_)*<regrmodel>.<eval-part>.(<predr>_)*<signif-type>.signif
%.signif:  bin/signif-$$(lastword $$(subst _, ,$$(suffix $$*)))  \
           $$(basename $$(basename $$(basename %))).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))$$(subst .,_-,$$(basename $$(subst _,.,$$(suffix $$*))))_$$(lastword $$(subst _, ,$$(basename $$(basename $$*))))$$(suffix $$(basename $$*))_predicted.errors  \
           $$(basename $$(basename $$(basename %))).$$(subst .,_,$$(basename $$(subst _,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))))$$(subst .,_,$$(basename $$(subst _,.,$$(suffix $$*))))_$$(lastword $$(subst _, ,$$(basename $$(basename $$*))))$$(suffix $$(basename $$*))_predicted.errors
	$^  >  $@


