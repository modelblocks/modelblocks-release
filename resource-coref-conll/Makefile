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

SHELL := /bin/bash

#### preamble...
.SUFFIXES:
.SECONDEXPANSION:

#### directories used by included items...
BIN := bin
GENMODEL := genmodel

#### included make items...
# include $(dir $(CURDIR))resource-general/Makefile
# include $(RESOURCE-LTREES)/Makefile
# include $(RESOURCE-GCG)/Makefile
# include $(dir $(CURDIR))coref/Makefile
# include $(RESOURCE-LVPCFG)/Makefile
# include $(RESOURCE-LOGREG)/Makefile
# include $(RESOURCE-TREEBANK)/Makefile
# include $(RESOURCE-LCPARSE)/Makefile
# include $(RESOURCE-CONLLSCORER)/Makefile
#include $(dir $(CURDIR))resource-wordnet/Makefile

#INCLUDE := $(abspath $(RESOURCE-INCRSEM)/include)
#SRC := $(abspath $(RESOURCE-INCRSEM)/src)

ifndef MAKED
$(info Building source code dependencies using recursive sub-make to $(RESOURCE-COREF-CONLL)/Makefile...)
## create dependencies for files in python, src, include directories
DUMMY := $(shell $(MAKE) MAKED=1 MAKECONFIG=1 $(subst .py,.d,$(wildcard $(CONLL-SCRIPTS)/*.py)) $(subst .cpp,.d,$(wildcard $(SRC)/*.cpp)))
$(info Source code dependencies built and loaded.)
endif

## include those dependencies
include $(wildcard $(RESOURCE-COREF-CONLL)/*/*.d)

################################################################################
#
#  i. Code compilation items
#
################################################################################

ifdef MAKED
#### python dependencies
.PRECIOUS:  $(CONLL-SCRIPTS)/%.d
$(CONLL-SCRIPTS)/%.d:  $(CONLL-SCRIPTS)/%.py
	echo '$@: \'  >  $@   #' ##
	grep 'import' $<  |  sed 's/^from \(.*\) import .*/$$(wildcard $$(RESOURCE-COREF-CONLL)\/..\/\*\/scripts\/\1.py) \\/;s/^import \(.*\)$$/$$(wildcard $$(RESOURCE-COREF-CONLL)\/..\/\*\/scripts\/\1.py) \\/'  >>  $@
	echo ''  >>  $@
	cat $@  |  sed 's/\.d:/\.py:/'  >>  $@
endif

################################################################################

%.key.conll: $(CONLL-SCRIPTS)/numberededitabletrees2conllcoref.py %.fromlinetrees.numbered.editabletrees
	python3 $< $(word 2, $^) > $@

%.preds.conll: $(CONLL-SCRIPTS)/tokdecs2conllcoref.py %.tokdecs 
	python3 $< $(word 2, $^) > $@


