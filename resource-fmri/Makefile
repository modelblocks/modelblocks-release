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

################################################################################
#
#  Combined toks file creation
#
################################################################################

%.hrf.itemmeasures: $$(basename %).t.itemmeasures $(FMRI-SCRIPTS)/hrf_convolve_predictors.py scripts/$$(subst .,,$$(suffix $$*)).pred_list.txt
	python $(word 2, $^) $(word 1, $^) --start $(SAMPLE-START) --predictors $(shell cat $(word 3, $^)) > $@

.PRECIOUS: %.core.evmeasures
%.core.evmeasures: %.itemmeasures $$(word 1, $$(subst ., , %)).src.evmeasures \
$(RT-SCRIPTS)/merge_tables.py
	cat $< | python $(word 3, $^) $< $(word 2, $^) sampleid > $@

