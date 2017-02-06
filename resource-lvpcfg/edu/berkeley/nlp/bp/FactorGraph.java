package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.util.Iterables;
import edu.berkeley.nlp.util.Iterators;

import java.util.List;
import java.util.ArrayList;

/**
 * User: aria42
 * Date: Jan 24, 2009
 */
public class FactorGraph {
  public List<Variable> vars;
  public List<Factor> factors;
  private boolean locked = false;

  public FactorGraph(Iterable<Variable> vars0) {
    vars = Iterators.fillList(vars0.iterator());
    for (int i = 0; i < vars.size(); i++) {
      Variable var = vars.get(i);
      var.index = i;
    }
    factors = new ArrayList<Factor>();
  }

  public void addFactor(List<? extends  Variable> inputs0, FactorPotential fp) {
    if (locked) throw new RuntimeException("Can't add to locked FG");
    List<Variable> inputs = new ArrayList<Variable>(inputs0);
    for (Variable v: inputs) {
      if (v.factors == null) {
        v.factors = new ArrayList<Factor>();
      }
    }
    Factor factor = new Factor(factors.size(),fp,inputs);
    for (Variable var: inputs) {
      var.factors.add(factor);
    }
    factors.add(factor);
  }

  public void lock() {
    // Don't do anything if we
    // already locked
    if (locked) return;
    locked = true;
    for (Variable var : vars) {
      int[] neighborIndices = new int[var.factors.size()];
      var.marginals = new double[var.numVals];
      for (int i = 0; i < var.factors.size(); i++) {
        neighborIndices[i] = var.factors.get(i).vars.indexOf(var);
        assert neighborIndices[i] >= 0;
      }
      var.neighborIndices = neighborIndices;
    }
    for (Factor f : factors) {
      int[] neighborIndices = new int[f.vars.size()];
      for (int i = 0; i < f.vars.size(); i++) {
        neighborIndices[i] = f.vars.get(i).factors.indexOf(f);
        assert neighborIndices[i] >= 0;
      }
      f.neighborIndices = neighborIndices;
    }
  }
}
