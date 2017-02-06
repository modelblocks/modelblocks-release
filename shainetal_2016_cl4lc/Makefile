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
#  Default recipe
#
################################################################################

default: DEV-ALL

################################################################################
#
#  Includes
#
################################################################################

include $(dir $(CURDIR))resource-general/Makefile
include $(RESOURCE-TOKENIZER)/Makefile
include $(RESOURCE-LTREES)/Makefile
include $(RESOURCE-GCG)/Makefile
include $(RESOURCE-RT)/Makefile
include $(RESOURCE-LVPCFG)/Makefile
include $(RESOURCE-LCPARSE)/Makefile
include $(RESOURCE-BNC)/Makefile
include $(RESOURCE-TREEBANK)/Makefile
include $(RESOURCE-GIGAWORD)/Makefile
include $(RESOURCE-KENLM)/Makefile
include $(RESOURCE-SRILM)/Makefile
include $(RESOURCE-INCRSEM)/Makefile
include $(RESOURCE-NATSTORIES)/Makefile
include $(RESOURCE-CHILDES)/Makefile
include $(RESOURCE-RT)/Makefile
include $(RESOURCE-LMEFIT)/Makefile

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

VPATH := $(RESOURCE-RT)

.SUFFIXES:
.SECONDEXPANSION:

LMEDEFAULT := NSCFX
# Buggy published exploratory/confirmatory split:
BASENAME := genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm-iz.fp.o.spr
# More sensible non-published exploratory/confurmatory split:
# BASENAME := genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm.fp.spr


################################################################################
#
#  Experiments
#
################################################################################

$(eval $(RT-EXPERIMENTS))

# All results published in CL4LC 2016 paper on Natural Stories
cl4lc2016: DEV-ALL TEST-BEST

# Published results on exploratory corpus
DEV-ALL: MEM-ALLSPILL

# Published results on confirmatory corpus
TEST-BEST: noF-S1_NSCFXt reinst-S1_NSCFXt dltcm-S1_NSCFXt dlt-S1_NSCFXt

# Diamond evaluation of fjprob vs. dlt (CoordMod) as predictors of SPR times.
noFS1-v-dltcmS1: genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm-iz.fp.o.spr.dltcmS1_noFS1.-NSCFX..diamond.lrtsignif

# Error analysis with certain events filtered out
ERROR-ANALYSIS: \
genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm-iz.fp.o.spr.dltcmS1.-NSCFX_-R_noVNposS1..lrtsignif \
genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm-iz.fp.o.spr.noFS1.-NSCFX_-R_noVNAdPrimS2..lrtsignif \
genmodel/naturalstories.wsj02to21-gcg15-decoupled-fg-3sm-bd.x-efabp.-c_-b5000.5-kenlm-iz.fp.o.spr.noFS1.-NSCFX_-R_noVNposS1..lrtsignif

