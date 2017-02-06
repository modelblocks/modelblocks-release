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
#  Templates for recipes in this directory:
#
#  - Automatically run an experiment from a config file:
#
#      To run an experiment from a config file named
#      <path>/<sents-basename>_<uhhmm-param-descr>.ini, type
#
#      make <path>/<sents-basename>_<uhhmm-param-descr>.uhhmm
#
#      This does not actually make the target file, so it will
#      always run. Make needs recipes to generate <sents-basename>.txt,
#      a sents file to input to the parser, so if that is not available
#      here already you may need to include another resource-* Makefile
#      containing the needed recipe(s).
#
#
#  - Automatically generate logprob and recall curves from experiment
#    output:
#
#      To generate synevals for all samples in an experiment directory,
#      then plot the logprob curve and extract and plot the recall curve,
#      type
#
#      make <path>/<gold-linetrees-basename>.synevals
#
#      As above, <gold-linetrees-basename>.linetrees must exist or be
#      buildable from an included recipe, so you may need to add your own
#      include statement(s) when setting up experiments on new corpora.
#      
#      Of course, the recipe above will only work if there are one or more
#      UHHMM last_sample*.txt files in the directory you point to.
#
################################################################################

################################################################################
#
#  Default
#
################################################################################

# Generates the parsevals from all competing systems (and right-branching baseline)
# as presented in Shain et al. (2016).
coling2016: \
genmodel/eve.eve-nt-lower-nounary-ccl.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-upparse.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-45-10-dmv.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-45-10-u-dmv.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-rb.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-nopunc-ccl.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-nopunc-upparse.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-nopunc-45-10-dmv.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-nopunc-45-10-u-dmv.induc.constiteval.txt \
genmodel/eve.eve-nt-lower-nounary-nopunc-rb.induc.constiteval.txt

################################################################################
#
#  Includes
#
################################################################################

include $(dir $(CURDIR))resource-general/Makefile
include $(RESOURCE-LTREES)/Makefile
include $(RESOURCE-LVPCFG)/Makefile
include $(RESOURCE-LCPARSE)/Makefile
include $(RESOURCE-GCG)/Makefile
include $(RESOURCE-CHILDES)/Makefile
include $(RESOURCE-TREEBANK)/Makefile
include $(RESOURCE-CCL)/Makefile
include $(RESOURCE-UPPARSE)/Makefile
include $(RESOURCE-BMMM)/Makefile
include $(RESOURCE-DMV)/Makefile

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

