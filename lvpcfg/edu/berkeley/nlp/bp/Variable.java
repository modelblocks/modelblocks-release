package edu.berkeley.nlp.bp;

import java.util.List;


/**
 * User: aria42
 */
public class Variable {

  // You provide two below
  public Object id;
  public int numVals;

  // This is written by BP
  public double[] marginals;

  // All Below Fields Filled In For You
  List<Factor> factors;
  int index;
  // index of this variable for each factor
  // filled in after FactorGraph locked
  // For factors.get(i), neighborIndices[i]
  // is the index of the variable in the
  // factors list of vars
  int[] neighborIndices;

  public Variable(Object id, int numVals) {
    this.id = id;
    this.numVals = numVals;
  }
}
