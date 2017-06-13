README for ModelBlocks
======================

This is the `Modelblocks` software package.  It includes several
resources for constructing and evaluating broad-coverage probabilistic
models of cognitive processes, organized into projects centered around
different tasks and data sets.

Quickstart
----------
To quickly get started using modelblocks, read through the quickstart
guide in the neighboring `QUICKSTART.md` file. Below is more info about
the structure of Modelblocks.

Use of Makefiles
----------------
Each project exists in a subdirectory of the main modelblocks
directory.  In order to ensure the reproducibility of experiments
conducted using this resource, these project directories each contain
a Makefile, which specifies how data sets, output files, and
evaluation results are constructed. The repository contains several
`resource-XXX` repositories that contain reusable libraries of code,
project repositories that contain recipes for reproducing published
experimental results, and a `workspace` for experimentation and
development. Recipes will fail unless all dependencies to external
resources (text corpora, experimental data, external code libraries, etc.)
are satisfied. If you attempt to make a recipe that has a missing
dependency, Make will exit with an error message about which dependency
is missing and how you can access it.

For sandboxing and development, nearly all
ModelBlocks recipes can be created from a single workspace. To initialize
your workspace, simply type `make` at the repository root, then
navigate to the `workspace` directory. To reproduce a published experiment,
navigate to the relevant experiment directory (named by author/year)
and type make. NOTE: We do not guarantee indefinite future support of all
published results recipes. In some cases it may be necessary to revert
the repository to some previous state in order to reproduce a result. If
you are encountering errors as you try to reproduce a result, please contact
the ModelBlocks development team.

Documentation for the targets provided by each Makefile is provided
in the `docs` directory of this repository. For help building targets
for common recipes, you can use the browser-based ModelBlocks Assistant
tool. This can be created locally by navigating to the `docs` directory
of ModelBlocks and typing `make mbassist`. A remote version can also
be accessed online at http://go.osu.edu/mbassist.

Included Resources
------------------
Modelblocks makes use of several third-party data and software
resources.  Where licenses permit, these have been included directly
in the modelblocks package, so as to avoid version compatibility
issues and thereby ensure reproducibility.  In some cases open-source
software has been modified so as to produce a common data file format
required by other software.  All resources included in this package
are distributed under the Gnu General Public License (see LICENSE file
in this directory).

Included resources are described in `RESOURCES.md`
and denoted by a lack of [+] next to their name.

External Resources and `USER-*.TXT` Files
-----------------------------------------
When licenses of resources used in Modelblocks projects do not permit
redistribution, or when (usually data) resources are too large to be
included, the Makefile will generate an appropriately-named
`user-*.txt` configuration file in `modelblocks-repository/config/`,
in which a user may specify a path to an external copy of the resource.
When make is first invoked, Modelblocks will create an incorrect
pointer for each configuration file in the dependency chain, along with
console output indicating which configuration files are needed to create 
the recipe. Before re-running make, the needed third-party resources 
will need to be downloaded and the pointers updated in 
modelblocks-repository/config. This use of `user-*.txt` files is 
intended to allow users to specify external resources or other user-
specific data without having to modify the Makefile, which may be 
overwritten in subsequent updates to Modelblocks. 

External resources are described in `RESOURCES.md`
and denoted by a [+] next to their name. `RESOURCES.md` also specifies which
`user-*.txt` files are associated with each external resource.
