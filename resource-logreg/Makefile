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

# Only use the following directory variables in prerequisites.
# Otherwise, they may break when this makefile is called by another.
INCLUDE  := $(RESOURCE-LOGREG)/include
BIN := bin
GENMODEL := genmodel

#### include dependencies of .hpp and .cpp files
include $(wildcard $(RESOURCE-LOGREG)*/*.d)

################################################################################
#
#  Code compilation items
#
################################################################################

#SHELL    := /bin/bash
#INCLUDES := -I /usr/include/libxml2/ -I$(INCLUDE) -I$(abspath $(MBDIR)/resource-rvtl)
#CFLAGS   := $(INCLUDES) -Wall `cat $(CONFIGDIR)/user-cflags.txt` -fpermissive
#CC       := g++

#### c++ dependencies
.PRECIOUS: $(RESOURCE-LOGREG)%.d
$(RESOURCE-LOGREG)%.d:  $(RESOURCE-LOGREG)%.cpp  |  $(INCLUDE)
	echo '$@: \' > $@   #' ##
	echo `g++ -I$| -std=c++11 -MM $<` | sed 's/^.*:.*\.cpp */ $$(wildcard /;s/\\ *//g;s/$$/)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@

#### c++ executables
.PRECIOUS: $(BIN)/%
$(BIN)/%:  $(RESOURCE-LOGREG)/src/$$(notdir $$@).cpp  $(RESOURCE-LOGREG)/src/$$(notdir $$@).d  $(CONFIGDIR)/user-cflags.txt  |  $(BIN)  $(INCLUDE)
	g++ -I/usr/include/libxml2/ -I$(word 2,$|) -I$(abspath $(MBDIR)/resource-rvtl) -Wall `cat $(word 3,$^)` -fpermissive -std=c++11 $< -lm -larmadillo -lmlpack -lpthread -o $@

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

#### obtain logistic regression model from predictor-result training data
.PRECIOUS: %blrmodel
%blrmodel:  bin/blr-lbfgs  %blrdecpars
	cat $(word 2,$^)  |  $(word 1,$^) 0.1  >  $@

#### obtain multinomial logistic regression (maxent/softmax) model from predictor-result training data
.PRECIOUS: %mlrmodel
%mlrmodel:  bin/mlr-lbfgs  $$(basename %).$$(lastword $$(subst _, ,$$*))mlrdecpars  |  $(CONFIGDIR)/user-numthreads.txt
	cat $(word 2,$^)  |  $(word 1,$^) $(shell cat $|) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@  2>  $@.log

#### obtain multinomial logistic regression (maxent/softmax) model from predictor-result training data
.PRECIOUS: %sgdmlrmodel
%sgdmlrmodel:  bin/mlr-sgd  $$(basename %).$$(lastword $$(subst _, ,$$*))mlrdecpars  |  $(CONFIGDIR)/user-numthreads.txt
	cat $(word 2,$^)  |  $(word 1,$^) $(shell cat $|) $(subst _, ,$(subst .,,$(suffix $*)))  >  $@

