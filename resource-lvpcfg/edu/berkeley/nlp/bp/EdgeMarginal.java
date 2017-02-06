package edu.berkeley.nlp.bp;

/**
 * User: aria42
 * Date: Jan 26, 2009
 */
public class EdgeMarginal {
  public final Variable x, y;
  public final double[][] marginals;
  public EdgeMarginal(Variable x, Variable y, double[][] marginals) {
    this.x = x;
    this.y = y;
    this.marginals = marginals;
  }
}
