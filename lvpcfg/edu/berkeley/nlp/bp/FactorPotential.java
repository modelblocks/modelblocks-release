package edu.berkeley.nlp.bp;

/**
 * User: aria42
 */
public interface FactorPotential {
  /**
   * 
   * @param inputMessages  input variable messages
   * @param outputMessages  output variable message
   */
  public void   computeLogMessages(double[][] inputMessages, double[][] outputMessages);

  /**
   * Compute the marginal given the input variable converged messages. The
   * return type is an Object b/c depending on the arity of the FactorPotential
   * you get a double[], double[][], double[][][], etc...
   * 
   * @param inputMessages
   * @return
   */
  public Object computeMarginal(double[][] inputMessages);
}
