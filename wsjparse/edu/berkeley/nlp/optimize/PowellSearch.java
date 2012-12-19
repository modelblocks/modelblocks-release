package edu.berkeley.nlp.optimize;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.Function;

public class PowellSearch implements FunctionMinimizer {

	public double[] minimize(Function fn, double[] p, double tol) {
		int n = fn.dimension();
		double[] pt = DoubleArrays.clone(p);
		double[] ptt = new double[n];
		double[][] xi = getStandardBasis(n);		
		double[] xit  = new double[n];
		double fp = fn.valueAt(p);
		double lastValue = fp;
		for (int iter=0; iter < 10; ++iter) {
			int bigI = -1;
			double del = 0.0;
			double lineMin = fp;
			for (int i=0; i < n; ++i) {
				DoubleArrays.assign(xit,xi[i]);
				double fptt = lastValue;
				lineMin = lineMin(fn, p, xit);
				if (fptt - lineMin > del) {
					del = fptt - del;
					bigI = i;
				}
			}
			if (2.0 * (fp-lineMin) <= tol * (Math.abs(fp) + Math.abs(lineMin))) {
				return p;
			}
			for (int i=0; i < n; ++i) {
				ptt[i] = 2.0 * p[i] - pt[i];
				xit[i] = p[i] - pt[i];
				pt[i] = p[i];
			}
			double fptt = fn.valueAt( ptt );
			if (fptt < fp) {				
				double t=2.0*(fp-2.0*(lineMin)+fptt)*Math.sqrt(fp-(lineMin)-del)-del*Math.sqrt(fp-fptt);
				if (t < 0.0) {
					lineMin = lineMin(fn, p, xit);
					for (int i=0; i < n; ++i) {
						xi[i][bigI] = xi[i][n];						 
						xi[i][n]=xit[i];
					}
				}
			}
		}
		return p;
	}

	private double[][] getStandardBasis(int n) {
		double[][] xi = new double[n][n];
		for (int i=0; i < n; ++i) {
			xi[i][i] = 1.0;
		}
		return xi;
	}
	
	private double lineMin(Function fn, double[] p, double[] xi) {
		return 0.0;
	}

}
