package edu.berkeley.nlp.optimize;

import edu.berkeley.nlp.math.Function;

public interface FunctionMinimizer {
	public double[] minimize(Function fn, double[] initialX, double tolerance);
}
