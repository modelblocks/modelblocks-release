package edu.berkeley.nlp.math;

import edu.berkeley.nlp.util.Logger;

public class SubgradientMinimizer implements GradientMinimizer {

	int minIterations = -1;
	double initialStepSizeMultiplier = 0.01;
	double stepSizeMultiplier = .5;
	double EPS = 1e-10;
	int maxIterations = 2000;

	public double[] minimize(DifferentiableFunction function, double[] initial, double tolerance, boolean project){
		return null;
	}

	public double[] minimize(DifferentiableFunction function, double[] initial,
			double tolerance) {
		boolean printProgress = true;
		BacktrackingLineSearcher lineSearcher = new BacktrackingLineSearcher();
		double[] guess = DoubleArrays.clone(initial);
		for (int iteration = 0; iteration < maxIterations; iteration++) {
			double[] subgradient = function.derivativeAt(guess);
			double value = function.valueAt(guess);
			double[] direction = subgradient;
			DoubleArrays.scale(direction, -1.0);
			if (iteration == 0) lineSearcher.stepSizeMultiplier = initialStepSizeMultiplier;
			else lineSearcher.stepSizeMultiplier = stepSizeMultiplier;
			double[] nextGuess = lineSearcher.minimize(function, guess,
					direction);
			double[] nextDerivative = function.derivativeAt(nextGuess);
			double nextValue = function.valueAt(nextGuess);
			if (printProgress) {
				Logger.i().logs("[Subgradient] Iteration %d: %.6f", iteration, nextValue);
			}

			if (iteration >= minIterations
					&& converged(value, nextValue, tolerance)) {
				return nextGuess;
			}
			guess = nextGuess;
			value = nextValue;
			subgradient = nextDerivative;
		}
		return guess;
	}

	private boolean converged(double value, double nextValue, double tolerance) {
		if (value == nextValue) return true;
		double valueChange = Math.abs(nextValue - value);
		double valueAverage = Math.abs(nextValue + value + EPS) / 2.0;
		if (valueChange / valueAverage < tolerance) return true;
		return false;
	}

	public void setMaxIterations(int maxIterations2) {
		maxIterations = maxIterations2;
	}

	public void setMinIteratons(int minIterations2) {
		minIterations = minIterations2;
	}
}
