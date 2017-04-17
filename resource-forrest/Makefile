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
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-RT)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

VPATH += $(RESOURCE-FORREST)

#SAMPLE-START := 20.05891632653599
FORREST-SUBJECTS := 001 002 003 004 005 006 007 008 009 011 012 013 014 015 016 017 018 019 020
# NOTE: Subject 010 doesn't have atlas-aligned data, and so was omitted from analysis

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-forrest-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
FORRESTDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(FORRESTDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(FORRESTDIR)$(MSG3))
endif
endif

%/user-forrest-directory.txt: | %   
	echo '/home/corpora/original/german/forrest_gump_audio' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

genmodel/forrest.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/forrest.rm_unfix_items.params: | genmodel
	echo '-f fdur' > $@

genmodel/forrest.time.tokmeasures: $(FORREST-SCRIPTS)/annotations2tokmeasures.py $(FORRESTDIR)/stimulus/task001/annotations/german_audio_description.csv | genmodel
	python $(word 1, $^) $(word 2, $^) > $@

genmodel/forrest.linetoks: $(FORREST-SCRIPTS)/tokmeasures2linetoks.py genmodel/forrest.time.tokmeasures $(TOKENIZERDIR)/ptb_tokenizer.sed | genmodel
	cat $(word 2, $^) | python $(word 1, $^) | $(word 3, $^) > $@

genmodel/forrest.lineitems: $(FORREST-SCRIPTS)/tokmeasures2linetoks.py genmodel/forrest.time.tokmeasures | genmodel
	cat $(word 2, $^) | python $(word 1, $^) > $@

genmodel/forrest.src.evmeasures: $(CONFIGDIR)/user-forrest-directory.txt $(FORREST-SCRIPTS)/compile_forrest_subjects.py $(foreach s, $(FORREST-SUBJECTS), $(foreach r, $(shell seq 1 8), $(FORRESTDIR)/phase1/sub$(s)/BOLD/task001_run00$(r)/atlas_timeseries_dico7Tad2grpbold7Tad_nl.csv.gz)) | genmodel
	python $(word 2, $^) 001 $(foreach r, $(shell seq 1 8),$(FORRESTDIR)/phase1/sub001/BOLD/task001_run00$(r)/atlas_timeseries_dico7Tad2grpbold7Tad_nl.csv.gz) > $@
	$(foreach s, $(FORREST-SUBJECTS), python $(word 2, $^) $(s) $(foreach r, $(shell seq 1 8),$(FORRESTDIR)/phase1/sub$(s)/BOLD/task001_run00$(r)/atlas_timeseries_dico7Tad2grpbold7Tad_nl.csv.gz) | tail -n+2 >> $@; )

#genmodel/forrest.generic.evmeasures: $(CONFIGDIR)/user-forrest-directory.txt $(FORRESTDIR)/forrest_data_shared/sept18-generic-predictors-under-hrf-with-logfrqhal.txt | genmodel/
#	echo 'rateConv frqConvOrth breakConvOrth tdpConvOrth tdxConvOrth bupConvOrth buxConvOrth cfgsurpConvOrth bigramlexConvOrth trigramlexConvOrth bigramposConvOrth trigramposConvOrth' > $@.tmp
#	tail -n+11 $(word 2, $^) | sed 's/\r//g;' | sed 's/\t/ /g;' >> $@.tmp
#	paste -d' ' $@.tmp <(echo 'sampleid'; for i in {1..362}; do echo "$$i"; done;) > $@
#	rm -f $@.tmp

#genmodel/forrest.src.evmeasures: genmodel/forrest.sbj.evmeasures genmodel/forrest.generic.evmeasures $(RESOURCE-RT)/scripts/merge_tables.py | genmodel
#	python $(word 3, $^) $(word 1, $^) $(word 2, $^) sampleid > $@
