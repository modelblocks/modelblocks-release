package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.Logger;

import java.util.List;

/**
 * User: aria42
 * Date: Jan 24, 2009
 */
public class BeliefPropogation {

  // fv = Factor to Variable Log Messages
  // [factor-index][neighbor-index][values]
  private double[][][] fv;
  // vf = Variable to Factor Log Messages
  // [var-index][neighbor-index][values]
  private double[][][] vf;
  // FactorGraph fg
  private FactorGraph fg;

  // Running Options
  private double tolerance = 0.0001;
  private int maxIterations = 10;
  private boolean verbose = false;
  private boolean debug = true;

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  public void setTolerance(double tolerance) {
    this.tolerance = tolerance;
  }

  public void run(FactorGraph fg) {
    init(fg);    
    for (int iter = 0; iter < maxIterations; iter++) {
      updateVariableToFactor();
      updateFactorToVariable();            
      double maxDiff = doVariableMarginals();
      if (verbose) Logger.logs("[BP] After %d iters, max change in var marginals=%.5f\n",iter+1,maxDiff);
      if (maxDiff < tolerance) {
        break;
      }
    }
    doFactorMarginals();    
  }

  private void doFactorMarginals() {
    for (int m = 0; m < fg.factors.size(); m++) {
      Factor f = fg.factors.get(m);
      double[][] varToFactorMessages = collectFactorMessage(f);
      f.marginals = f.potential.computeMarginal(varToFactorMessages);
    }    
  }

  private double doVariableMarginals() {
    double maxDiff = Double.NEGATIVE_INFINITY;
    for (int n = 0; n < fg.vars.size(); n++) {
      Variable v = fg.vars.get(n);
      double[] marginals = new double[v.numVals];
      for (int m=0; m < v.factors.size(); ++m) {
        Factor f = v.factors.get(m);
        int factorIndex = f.index;
        int varIndex = v.neighborIndices[m];
        DoubleArrays.addInPlace(marginals,fv[factorIndex][varIndex]);
      }
      SloppyMath.logNormalize(marginals);
      marginals = DoubleArrays.exponentiate(marginals);
      maxDiff = Math.max(maxDiff,DoubleArrays.lInfinityDist(marginals,v.marginals));
      v.marginals = marginals;
    }
    return maxDiff;
  }

  private double[][] collectFactorMessage(Factor f) {
    double[][] localVF = new double[f.vars.size()][];
    for (int n = 0; n < f.vars.size(); n++) {
      int varIndex = f.vars.get(n).index;
      int neighborIndex = f.neighborIndices[n];
      localVF[n] = vf[varIndex][neighborIndex];
    }
    return localVF;
  }

  private void updateFactorToVariable() {
    for (int m = 0; m < fg.factors.size(); m++) {
      Factor f = fg.factors.get(m);
      double[][] localVF = collectFactorMessage(f);      
      f.potential.computeLogMessages(localVF, fv[m]);
      if (debug) DoubleArrays.checkValid(fv[m]);
    }
  }

  private void updateVariableToFactor() {
    for (int n = 0; n < fg.vars.size(); n++) {
      Variable var = fg.vars.get(n);
      int D = var.numVals;      
      double[] sums = new double[D];
      for (int m = 0; m < var.factors.size(); m++) {
        Factor f = var.factors.get(m);
        int factorIndex = f.index;
        int varIndex = var.neighborIndices[m];
        DoubleArrays.addInPlace(sums, fv[factorIndex][varIndex]);
      }      
      for (int m = 0; m < var.factors.size(); m++) {
        Factor f = var.factors.get(m);
        int factorIndex = f.index;
        int varIndex = var.neighborIndices[m];
        DoubleArrays.assign(vf[n][m],sums);
        DoubleArrays.subtractInPlaceUnsafe(vf[n][m], fv[factorIndex][varIndex]);
        SloppyMath.logNormalize(vf[n][m]);
        if (debug) DoubleArrays.checkValid(vf[n][m]);
      }
    }
  }

  private void init(FactorGraph fg) {
    this.fg =  fg;
    this.fg.lock();
    this.fv = makeFactorToVariableMessages();
    this.vf = makeVariableToFactorMessages();
  }

  private double[][][] makeVariableToFactorMessages() {
    int N = fg.vars.size();
    double[][][] vf = new double[N][][];
    for (int n = 0; n < fg.vars.size(); n++) {
      Variable var = fg.vars.get(n);
      List<Factor> factors = var.factors;
      vf[n] = new double[factors.size()][var.numVals];
      for (double[] row : vf[n]) {
        SloppyMath.logNormalize(row);
      }      
    }
    return vf;
  }

  private double[][][] makeFactorToVariableMessages() {
    int M = fg.factors.size();
    double[][][] fv = new double[M][][];
    for (int m = 0; m < M; m++) {
      Factor f = fg.factors.get(m);
      fv[m] = new double[f.vars.size()][];
      for (int n = 0; n < f.vars.size(); n++) {
        Variable var = f.vars.get(n);
        fv[m][n] = new double[var.numVals];
        SloppyMath.logNormalize(fv[m][n]);
      }      
    }
    return fv;
  }
}
