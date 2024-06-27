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

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-glstm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GLSTMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GLSTMDIR))))
RESOURCE-NAME := Gulordava LSTM
define RESOURCE-DESCR =

NAME: Gulordava LSTM
MB POINTER FILE: config/user-glstm-directory.txt
AVAILABILITY: Free
URL: https://github.com/facebookresearch/colorlessgreenRNNs
DESCRIPTION: LSTM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.
endef
GLSTM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GLSTM = $(error $(GLSTM-ERROR))
else
DEP-GLSTM :=
endif
endif

%/user-glstm-directory.txt: | %   
	echo '/fs/project/lin-dept/compling/glstm' > $@

################################################################################

glstm_env: $(RESOURCE-GLSTM)/glstm_conda_env.sh
	$(SHELL) $(word 1, $^)

.PRECIOUS: %.glstm.tokmeasures
%.glstm.tokmeasures:  %.senttoks $(GLSTM-SCRIPTS)/generate_surprisal.py  |  $(CONFIGDIR)/user-glstm-directory.txt
	##################################################################
	### This target assumes active (glstm_env) conda environment #####
	### Please run "make glstm_env" and "conda activate glstm_env" ###
	##################################################################
	#export LD_LIBRARY_PATH=$(CONDA_PREFIX)/lib64
	python3 $(word 2,$^) --checkpoint $(shell cat $|)/hidden650_batch128_dropout0.2_lr20.0.pt --path $(word 1,$^) --data $(RESOURCE-GLSTM)/src >  $@  2>  $@.log

