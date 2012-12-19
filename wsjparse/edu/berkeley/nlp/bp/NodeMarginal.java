package edu.berkeley.nlp.bp;

/**
 * User: aria42
 * Date: Jan 26, 2009
 */
public class NodeMarginal {
  public final Variable x;
  public final double[] marginal;

  public NodeMarginal(Variable x, double[] marginal) {
    this.x = x;
    this.marginal = marginal;
  }
}
