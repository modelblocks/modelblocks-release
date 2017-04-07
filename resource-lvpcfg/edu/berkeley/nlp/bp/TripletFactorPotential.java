package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.math.SloppyMath;

/**
 * User: aria42
 * Date: Jan 27, 2009
 */
public class TripletFactorPotential implements FactorPotential {

  private double[][][] potentials;

  public TripletFactorPotential(double[][][] potentials) {
    this.potentials = potentials;
  }

  public void computeLogMessages(double[][] inputMessages, double[][] outputMessages) {
    int D1 = inputMessages[0].length;
    int D2 = inputMessages[1].length;
    int D3 = inputMessages[2].length;
    for (int d1 = 0; d1 < D1; d1++) {
      for (int d2 = 0; d2 < D2; d2++) {
        for (int d3 = 0; d3 < D3; d3++) {
          int[] vals = {d1,d2,d3};
          double p = potentials[d1][d2][d3];
          double sum = 0.0;
          for (int i=0; i < 3; ++i) {
            sum += inputMessages[i][vals[i]];
          }
          for (int i = 0; i < 3; i++) {
            int d = vals[i];
            double curVal = outputMessages[i][d];
            double diff = sum > Double.NEGATIVE_INFINITY ?
                sum - inputMessages[i][d] : sum;
            curVal = SloppyMath.logAdd(curVal,p + diff);
            outputMessages[i][d] = curVal;
          }
        }
      }
      for (int i=0; i < 3; ++i) SloppyMath.logNormalize(outputMessages[i]);
    }

  }

  public Object computeMarginal(double[][] inputMessages) {
    int D1 = inputMessages[0].length;
    int D2 = inputMessages[1].length;
    int D3 = inputMessages[2].length;
    double[][][] marginals = new double[D1][D2][D3];
    double[] pieces = new double[D1*D2*D3];
    int index = 0;
    for (int d1 = 0; d1 < D1; d1++) {
      for (int d2 = 0; d2 < D2; d2++) {
        for (int d3 = 0; d3 < D3; d3++) {
          int[] vals = {d1,d2,d3};
          double sum = potentials[d1][d2][d3];
          for (int i = 0; i < 3; i++) {
            sum += inputMessages[i][vals[i]];
          }
          pieces[index++] = sum;
          marginals[d1][d2][d3] = sum;
        }
      }
    }
    double logSum = SloppyMath.logAdd(pieces);
    for (int d1 = 0; d1 < D1; d1++) {
      for (int d2 = 0; d2 < D2; d2++) {
        for (int d3 = 0; d3 < D3; d3++) {
          marginals[d1][d2][d3] = Math.exp(marginals[d1][d2][d3] - logSum);
        }
      }
    }
    return marginals;
  }


}
