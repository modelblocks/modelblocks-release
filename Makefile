workspace: workspace_makefile_src.txt
	mkdir -p workspace
	cat $(word 1, $^) > workspace/Makefile

