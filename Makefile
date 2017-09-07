workspace: workspace_makefile_src.txt resource-general/Makefile
	mkdir -p workspace
	cat $(word 1, $^) > workspace/Makefile

RESOURCES.md: getResourceDescr.py $(wildcard resource-*/Makefile)
	python $^ > $@
