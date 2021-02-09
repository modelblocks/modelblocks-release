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


jlstm: bin
	cp -r $(JLSTM-SCRIPTS)/* $<
	cd $< && bazel build -c opt lm_1b/... && cd ..

.PRECIOUS: %.jlstm.tokmeasures jlstm
$(GENMODEL)/%.jlstm.tokmeasures:  %.delim.linetoks jlstm  |  $(CONFIGDIR)/user-jlstm-directory.txt
	bin/bazel-bin/lm_1b/lm_1b_eval --mode eval --pbtxt $(shell cat $|)/graph-2016-09-10.pbtxt \
	                               --vocab_file $(shell cat $|)/vocab-2016-09-10.txt  --input_data $(word 1,$^) \
                                   --ckpt '$(shell cat $|)/ckpt-*' > $@  2> $@.log
