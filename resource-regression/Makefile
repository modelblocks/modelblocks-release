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

################################################################################
#
#  Recipes
#
################################################################################

%.cens.resmeasures: %.resmeasures $(REGRESSION-SCRIPTS)/censor.py $(subst .,,$(suffix %)).cens_params.ini
	cat $< | python $(word 2, $^) $(word 3, $^) > $@

%.part.resmeasures: $(basename %).resmeasures $(REGRESSION-SCRIPTS)/partition.py $(word 1, $(subst ., ,%)).part_params.ini
	cat $< | python $(word 2, $^) -c $(word 3, $^) -p $(subst +, ,$(subst .,,$(suffix $*))) > $@
