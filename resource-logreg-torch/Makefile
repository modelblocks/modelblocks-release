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

#### default item: the main product of this directory...


################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

THISDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Only use the following directory variables in make item prerequisites.
# Otherwise, they may break when this makefile is called by another.
SCRIPTS := $(THISDIR)scripts


################################################################################
#
#  Local package installation
#
################################################################################

~/torch/install/bin/th:
	git clone https://github.com/torch/distro.git ~/torch --recursive
	cd ~/torch; ./install.sh
	~/torch/install/bin/luarocks install optim
	~/torch/install/bin/luarocks install penlight
	~/torch/install/bin/luarocks install rnn
	~/torch/install/bin/luarocks install tds


################################################################################
#
#  Model files
#
#  to construct the following file types:
#    <x>model  : predictor weights, one predictor per line, in format: '<predictorname> = <weight>'
#  given the following file types:
#    <x>decpars : comma-delimited predictor values and result, one result per line, in format: '<predictorname>=<value>,<predictorname>=<value>,... : <result>'
#
################################################################################

#### obtain multinomial logistic regression (maxent/softmax) model from predictor-result training data
.PRECIOUS: %mlrtorchmodel
%mlrtorchmodel:  ~/torch/install/bin/th  $(SCRIPTS)/maxent.lua  $$(basename %)$$(suffix $$(subst _,.,$$*))mlrdecpars
	$<  $(word 2,$^)  $(subst .,,$(subst ., ,$(basename $(subst _,.,$(suffix $*)))))  -trainingdata $(word 3,$^)  -lossdifference 1e-2  -outputfile $@


