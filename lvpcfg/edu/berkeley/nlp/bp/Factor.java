package edu.berkeley.nlp.bp;

import java.util.List;

/**
 * User: aria42
 */
public class Factor {
  int index;
  public FactorPotential potential;
  public List<Variable> vars;
  public Object marginals;
  int[] neighborIndices;

  Factor(int index, FactorPotential potential, List<Variable> vars) {
    this.index = index;
    this.potential = potential;
    this.vars = vars;
  }
}
