package edu.berkeley.nlp.optimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.math.Function;

public class GridSearch implements FunctionMinimizer {
	
	private double min, max, step;
	
	public GridSearch(double min, double max, double step) {
		this.min = min;
		this.max = max;
		this.step = step;
	}
	
	private int n ;
	private double bestValue = Double.POSITIVE_INFINITY;
	private double[] bestX;
	private double[] curX;
	private Function fn;
	
	public double[] minimize(Function fn, double[] initialX, double tolerance) {
		this.n = fn.dimension();
		this.bestX = new double[n];
		this.curX = new double[n];
		this.fn = fn;
		gridRecurse(new ArrayList<Double>());
		return bestX;
	}
		
	
	private void gridRecurse(List<Double> vals) {	
		if (vals.size() == n) {
			for (int i=0; i < n; ++i) {
				curX[i] = vals.get(i);
			}
			double val = fn.valueAt(curX);
			if (val < bestValue) {
				DoubleArrays.assign(bestX, curX);
				bestValue = val;
			}
			return;
		}
		for (double x=min; x <= max; x += step) {
			List<Double> newVals = new ArrayList<Double>(vals);
			newVals.add(x);
			gridRecurse(newVals);
		}
	}
	
	public static void main(String[] args) {
		Function fn = new Function() {

			public int dimension() {
				return 3;				
			}

			public double valueAt(double[] x) {
				System.out.println("x: " + Arrays.toString(x));
				return 0;
			}
			
		};
		GridSearch gridSearch = new GridSearch(1.0,5.0,1.0);
		gridSearch.minimize(fn, null, 1.0e-4);
		
	}
 
}
