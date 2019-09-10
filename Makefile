.SUFFIXES:
.SECONDEXPANSION:

workspace: workspace_makefile_src.txt resource-general/Makefile
	mkdir -p workspace
	cat $(word 1, $^) > workspace/Makefile

RESOURCES.md: getResourceDescr.py $(wildcard resource-*/Makefile)
	python $^ > $@

%.test.linetrees: $$(shell python -m mb.dep % linetrees -o .test)
	echo $@
