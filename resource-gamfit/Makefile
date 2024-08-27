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

#### copy gamform to local scripts directory...
scripts/%.gamform:  $(GAMFIT-SCRIPTS)/%.gamform  |  scripts
	cp $^ $@

bin/regress-gam: $(GAMFIT-SCRIPTS)/regress-gam.sh $(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R) | bin
	cp $< $@

bin/predict-gam: $(GAMFIT-SCRIPTS)/predict-gam.sh $(LMEFIT-SCRIPTS)/lmetools.r \
$(abspath $(RHACKS-SCRIPTS)/mer-utils.R) \
$(abspath $(RHACKS-SCRIPTS)/regression-utils.R) | bin
	cp $< $@
