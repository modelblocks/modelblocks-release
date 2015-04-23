package edu.berkeley.nlp.bp;

import edu.berkeley.nlp.util.CallbackFunction;
import edu.berkeley.nlp.util.functional.Function;
import edu.berkeley.nlp.math.SloppyMath;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * User: aria42
 * Date: Jan 27, 2009
 */
public class GenericFactorPotential implements FactorPotential {

    private int[] vars;
    private int[] scratch;
    private Function<int[],Double> potentialFn;

    public GenericFactorPotential(int[] vars, Function<int[],Double> potentialFn) {
      this.potentialFn = potentialFn;
      this.vars = new int[vars.length];
      this.scratch = new int[vars.length];
      System.arraycopy(vars,0,this.vars,0,vars.length);        
    }

    private void combinations(CallbackFunction cf) {
      combinations(cf,0);
    }

    private void combinations(CallbackFunction cf, int pos) {
      int D = vars[pos];
      for (int d = 0; d < D; d++) {
        scratch[pos] = d;
        if (pos+1 < vars.length) {
          combinations(cf,pos+1);
        } else {
          cf.callback(scratch);          
        }
      }        
    }

    public void computeLogMessages(final double[][] inputMessages, double[][] outputMessages) {
      for (int vx = 0; vx < vars.length; vx++) {
        final int v = vx;
        for (int dx = 0; dx < vars[v]; dx++) {
          final int d = dx;
          final List<Double> pieces = new ArrayList<Double>();
          combinations(new CallbackFunction() {
            public void callback(Object... args) {
              int[] assgn = (int[]) args[0];
              if (assgn[v] != d) return;
              double sum = potentialFn.apply(assgn);
              for (int vp=0; vp < vars.length; ++vp) {
                if (vp == v) continue;
                sum += inputMessages[v][assgn[v]];
              }
              pieces.add(sum);                  
            }
          });
          double logSum = SloppyMath.logAdd(pieces);
          outputMessages[v][d] = logSum;
        }
      }
    }

    public Object computeMarginal(double[][] inputMessages) {
        return null;
    }

}
