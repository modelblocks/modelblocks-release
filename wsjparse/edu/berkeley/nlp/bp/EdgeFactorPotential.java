package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.math.SloppyMath;

import fig.basic.NumUtils;

/**
 * User: aria42
 * Date: Jan 24, 2009
 */
public class EdgeFactorPotential implements FactorPotential {

  int D1, D2;
  double[][] logPotentials ;

  public EdgeFactorPotential(double[][] logPotentials) {
    this.logPotentials = NumUtils.copy(logPotentials);
    this.D1 = logPotentials.length;
    this.D2 = logPotentials[0].length;
  }

  public Object computeMarginal(double[][] inputMessages) {
    double[][] marginals = new double[D1][D2];
    double[] scratch = new double[D1*D2];
    int scratchIndex = 0;
    for (int d1 = 0; d1 < D1; d1++) {
      for (int d2 = 0; d2 < D2; d2++) {
        double logMarginal = logPotentials[d1][d2] +
              inputMessages[0][d1] +
              inputMessages[1][d2];
        marginals[d1][d2] = logMarginal;
        scratch[scratchIndex++] = logMarginal;
      }
    }
    double logSum = SloppyMath.logAdd(scratch);
    for (int d1 = 0; d1 < D1; d1++) {
      for (int d2 = 0; d2 < D2; d2++) {
        marginals[d1][d2] = Math.exp(marginals[d1][d2]-logSum);
      }
    }
    return marginals;
  }

  public void computeLogMessages(double[][] inputMessages, double[][] outputMessages) {
    for (int d1 =0; d1 < D1; ++d1) {      
      double[] pieces = new double[D2];
      for (int d2 = 0; d2 < D2; d2++) {
       pieces[d2] = (inputMessages[1][d2] + logPotentials[d1][d2]); 
      }
      outputMessages[0][d1] = SloppyMath.logAdd(pieces);
    }
    SloppyMath.logNormalize(outputMessages[0]);
    for (int d2 =0; d2 < D2; ++d2) {
      double[] pieces = new double[D1];
      for (int d1 = 0; d1 < D2; d1++) {
       pieces[d1] = (inputMessages[0][d1] + logPotentials[d1][d2]); 
      }
      outputMessages[1][d2] = SloppyMath.logAdd(pieces);
    }
    SloppyMath.logNormalize(outputMessages[1]);
  }
}
