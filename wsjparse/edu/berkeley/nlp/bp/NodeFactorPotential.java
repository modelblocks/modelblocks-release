package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.SloppyMath;

import java.util.List;
import java.util.ArrayList;

/**
 * User: aria42
 * Date: Jan 24, 2009
 */
public class NodeFactorPotential implements FactorPotential {

  private double[] potentials;
  private int D;

  public NodeFactorPotential(double[] potentials) {
    this.potentials = DoubleArrays.clone(potentials);
    this.D = this.potentials.length;
  }

  public void computeLogMessages(double[][] inputMessages, double[][] outputMessages) {
    DoubleArrays.assign(potentials,outputMessages[0]);
    SloppyMath.logNormalize(outputMessages[0]);
  }

  public Object computeMarginal(double[][] inputMessages) {
    double[] logMarginals = DoubleArrays.add(potentials,inputMessages[0]);
    SloppyMath.logNormalize(logMarginals);
    return DoubleArrays.exponentiate(logMarginals);
  }
}
