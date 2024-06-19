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
CONFIG := $(CONFIGDIR)/user-hf-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
HFDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(HFDIR))))
RESOURCE-NAME := HuggingFace LM
define RESOURCE-DESCR =

NAME: HuggingFace LM
MB POINTER FILE: config/user-hf-directory.txt
AVAILABILITY: Free
URL: https://huggingface.co/docs/transformers/en/index
DESCRIPTION: State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.
endef
HF-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-HF = $(error $(HF-ERROR))
else
DEP-HF :=
endif
endif

%/user-hf-directory.txt: | %
	echo '/project/lin-dept/compling/huggingface' > $@

################################################################################

hf_env: $(RESOURCE-HF)/hf_conda_env.sh
	$(SHELL) $(word 1, $^)

################################################################################
#
# GPT-2 (OpenAI): targets differently named from resource-gpt2 implementation
#
################################################################################

.PRECIOUS: %.gpt2.tokmeasures
%.gpt2.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) gpt2 token  >  $@  2>  $@.log

.PRECIOUS: %.gpt2-medium.tokmeasures
%.gpt2-medium.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) gpt2-medium token  >  $@  2>  $@.log

.PRECIOUS: %.gpt2-large.tokmeasures
%.gpt2-large.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) gpt2-large token  >  $@  2>  $@.log

.PRECIOUS: %.gpt2-xl.tokmeasures
%.gpt2-xl.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) gpt2-xl token  >  $@  2>  $@.log

################################################################################
#
# OPT (Meta)
#
################################################################################

.PRECIOUS: %.opt-125m.tokmeasures
%.opt-125m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-125m token  >  $@  2>  $@.log

.PRECIOUS: %.opt-350m.tokmeasures
%.opt-350m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-350m token  >  $@  2>  $@.log

.PRECIOUS: %.opt-1300m.tokmeasures
%.opt-1300m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-1.3b token  >  $@  2>  $@.log

.PRECIOUS: %.opt-2700m.tokmeasures
%.opt-2700m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-2.7b token  >  $@  2>  $@.log

%.opt-6700m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-6.7b token  >  $@  2>  $@.log

.PRECIOUS: %.opt-13000m.tokmeasures
%.opt-13000m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-13b token  >  $@  2>  $@.log

.PRECIOUS: %.opt-30000m.tokmeasures
%.opt-30000m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-30b token  >  $@  2>  $@.log

.PRECIOUS: %.opt-66000m.tokmeasures
%.opt-66000m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) facebook/opt-66b token  >  $@  2>  $@.log

################################################################################
#
# GPT-Neo (EleutherAI)
#
################################################################################

.PRECIOUS: %.gpt-neo-125m.tokmeasures
%.gpt-neo-125m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) EleutherAI/gpt-neo-125M token  >  $@  2>  $@.log

.PRECIOUS: %.gpt-neo-1300m.tokmeasures
%.gpt-neo-1300m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) EleutherAI/gpt-neo-1.3B token  >  $@  2>  $@.log

.PRECIOUS: %.gpt-neo-2700m.tokmeasures
%.gpt-neo-2700m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) EleutherAI/gpt-neo-2.7B token  >  $@  2>  $@.log

.PRECIOUS: %.gpt-j-6000m.tokmeasures
%.gpt-j-6000m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) EleutherAI/gpt-j-6B token  >  $@  2>  $@.log

.PRECIOUS: %.gpt-neox-20000m.tokmeasures
%.gpt-neox-20000m.tokmeasures:  %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $(word 2,$^) $(word 1,$^) EleutherAI/gpt-neox-20b token  >  $@  2>  $@.log

################################################################################
#
# Pythia (EleutherAI)
#
################################################################################

STEPS := 0 1 2 4 8 16 32 64 128 256 512 $(shell seq 1000 1000 143000)

define build_70m
%.pythia-70m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-70m step$(1) token  >  $$@  2>  $$@.log

endef

define build_160m
%.pythia-160m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-160m step$(1) token  >  $$@  2>  $$@.log

endef

define build_410m
%.pythia-410m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-410m step$(1) token  >  $$@  2>  $$@.log

endef

define build_1000m
%.pythia-1000m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-1b step$(1) token  >  $$@  2>  $$@.log

endef

define build_1400m
%.pythia-1400m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-1.4b step$(1) token  >  $$@  2>  $$@.log

endef

define build_2800m
%.pythia-2800m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-2.8b step$(1) token  >  $$@  2>  $$@.log

endef

define build_6900m
%.pythia-6900m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-6.9b step$(1) token  >  $$@  2>  $$@.log

endef

define build_12000m
%.pythia-12000m-step$(1).tokmeasures: %.sentitems $(HF-SCRIPTS)/get_llm_surprisal.py
	############################################################
	### This target assumes active (hf_env) conda environment ##
	### Please run "make hf_env" and "conda activate hf_env" ###
	############################################################
	python3 $$(word 2,$$^) $$(word 1,$$^) EleutherAI/pythia-12b step$(1) token  >  $$@  2>  $$@.log

endef

$(foreach i,$(STEPS),\
	$(eval $(call build_70m,$(i)))\
	$(eval $(call build_160m,$(i)))\
	$(eval $(call build_410m,$(i)))\
	$(eval $(call build_1000m,$(i)))\
	$(eval $(call build_1400m,$(i)))\
	$(eval $(call build_2800m,$(i)))\
	$(eval $(call build_6900m,$(i)))\
	$(eval $(call build_12000m,$(i)))\
)
