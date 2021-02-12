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

glstm_env: $(RESOURCE-GLSTM)/glstm_conda_env.sh
	$(SHELL) $(word 1, $^)

.PRECIOUS: %.glstm.tokmeasures
$(GENMODEL)/%.glstm.tokmeasures:  %.delim.linetoks $(GLSTM-SCRIPTS)/generate_surprisal.py  |  $(CONFIGDIR)/user-glstm-directory.txt
	##################################################################
	### This target assumes active (glstm_env) conda environment #####
	### Please run "make glstm_env" and "conda activate glstm_env" ###
	##################################################################
	python3 $(word 2,$^) --checkpoint $(shell cat $|) --path $(word 1,$^) --data $(RESOURCE-GLSTM)/src >  $@  2>  $@.log