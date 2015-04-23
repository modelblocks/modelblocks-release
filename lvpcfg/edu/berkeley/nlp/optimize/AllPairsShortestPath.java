package edu.berkeley.nlp.optimize;

public class AllPairsShortestPath {
	
	// Input must be a symmetric matrix, with edgeCosts[i][j] the cost of
	// the directed edge from node i to node j
	public double[][] getAllShortestPathCosts(double[][] edgeCosts) {
		if (edgeCosts == null) {
			return null;
		}
		if (edgeCosts.length == 0) {
			return new double[0][0];
		}
		if (edgeCosts.length != edgeCosts[0].length) {
			throw new IllegalArgumentException("Input must be a symmetric matrix");
		}
		int n = edgeCosts.length;
		double[][] costs = new double[n][n];
		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				if (i == j) {
					costs[i][j] = 0.0;
				} else {
					costs[i][j] = edgeCosts[i][j];
				}
			}
		}
		
		for (int k=0; k<n; k++) {
			for (int i=0; i<n; i++) {
				for (int j=0; j<n; j++) {
					costs[i][j] = Math.min(costs[i][j], costs[i][k] + costs[k][j]);
				}
			}
		}
		return costs;
	}
}
