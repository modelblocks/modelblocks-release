#!/bin/bash

# Prints the current git commit, F model ini, and J model ini for bookkeeping.

F_INI=/home/clark.3664/git/modelblocks-release/resource-incrsem/scripts/besttrf.ini
J_INI=/home/clark.3664/git/modelblocks-release/resource-incrsem/scripts/besttrj.ini

echo "==== F model INI ===="
cat $F_INI

echo $'\n\n'"==== J model INI ===="
cat $J_INI

echo $'\n\n'
git log -1

