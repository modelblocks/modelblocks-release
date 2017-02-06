package edu.berkeley.nlp.mt;

import java.util.List;

/**
 * @author Alexandre Bouchard
 * 
 * The sufficient statistics of a bleu score.
 *
 */
public class BleuScore {
	private List<Double> individualNGramScorings;
	private List<Double> weights;
	private double r;
	private double c;

	/**
	 * 
	 * @param individualNGramScoring The i-th position should be the i-gram modified precision
	 * @param weights The weigths used to combine the individual n-grams
	 * @param r The r value, as specified by BleuEvaluator
	 * @param c The c value, as specified by BleuEvaluator
	 */
	public BleuScore(List<Double> individualNGramScorings, List<Double> weights, double r,
			double c) {
		this.individualNGramScorings = individualNGramScorings;
		this.weights = weights;
		this.r = r;
		this.c = c;
	}

	public List<Double> getIndividualNGramScorings() {
		return individualNGramScorings;
	}

	/**
	 * 
	 */
	private double bleu = Double.NEGATIVE_INFINITY;

	/**
	 * 
	 * Compute (and cache) the geometric mean of the individual n grams and 
	 * multiply by the brevity penalty
	 * 
	 * @return
	 */
	public double getBleuScore() {
		if (bleu == Double.NEGATIVE_INFINITY) {
			double exponent = 0.0;
			for (int i = 0; i < individualNGramScorings.size(); i++) {
				exponent += Math.log(individualNGramScorings.get(i)) * weights.get(i);
			}
			bleu = brevityPenalty() * Math.exp(exponent);
			return bleu;
		} else {
			return bleu;
		}
	}

	/**
	 * 
	 * Long translations are already penalized by the precision metric, so 
	 * no further penalty are inflected if c > r, otherwise there is a 
	 * multiplicative penalty of exp(1 - r/c)
	 * 
	 * @return
	 */
	public double brevityPenalty() {
		if (c > r) {
			return 1.0;
		} else {
			return Math.exp(1 - r / c);
		}
	}

	/**
	 * 
	 * @return
	 */
	public String toString() {
		return new Double(getBleuScore()).toString();
	}

	/**
	 * 
	 * @return
	 */
	public double getScore() {
		return getBleuScore();
	}

	/**
	 * 
	 * @param arg0
	 * @return
	 */
	public int compareTo(Object arg0) {
		if (arg0 instanceof BleuScore) {
			BleuScore other = (BleuScore) arg0;
			if (this.getScore() < other.getScore()) {
				return -1;
			} else if (this.getScore() > other.getScore()) {
				return 1;
			} else {
				return 0;
			}
		} else {
			throw new ClassCastException();
		}
	}

	/**
	 * 
	 * @param arg0
	 * @return
	 */
	public boolean equals(Object arg0) {
		if (arg0 instanceof BleuScore) {
			BleuScore other = (BleuScore) arg0;
			if (this.getScore() == other.getScore()) {
				return true;
			}
		}
		return false;
	}



}
