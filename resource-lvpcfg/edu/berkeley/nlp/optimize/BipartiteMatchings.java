package edu.berkeley.nlp.optimize;

import fig.basic.BipartiteMatcher;
import fig.basic.NumUtils;
import edu.berkeley.nlp.math.DoubleArrays;

import java.util.Arrays;

public class BipartiteMatchings {

	private double[][] padMatrix(double[][] costMatrix) {
		int m = costMatrix.length;
		int n = costMatrix[0].length;
		int max = Math.max(m,n);
		int min = Math.min(m,n);
		double[][] padded = new double[max][max];
		for (int i=0; i < max; ++i) {
		  for (int j=0; j < max; ++j) {
			  if (i < m && j < n) {
		      padded[i][j] = i < m && j < n ? costMatrix[i][j] : Double.POSITIVE_INFINITY;
			  }
		  }
		}
		return padded;		     	
	}

	private int[] getPaddedMatching(int[] origMatching, int m, int n) {
		boolean paddedRows = m < n ;
		int min = Math.min(m,n);
		int[] paddedMatch = new int[m];
		if (paddedRows) {
			System.arraycopy(origMatching,0,paddedMatch,0,min);
			return paddedMatch;
		}
		for (int i=0; i < m; ++i) {
		  int j = origMatching[i];
			paddedMatch[i] = j < n ? j : -1;
		}
		return paddedMatch;
	}

	public int[] getMaxMatching(double[][] costMatrix) {
		int m = costMatrix.length;
		int n = costMatrix[0].length;
		if (m != n) {
			costMatrix = padMatrix(costMatrix);
		}
		int[] matching =  new BipartiteMatcher().findBestAssignment(costMatrix);
		return m != n ? getPaddedMatching(matching,m,n) : matching;
	}

	public double[][] getAllMaxMatchingCosts(double[][] originalCostMatrix) {
		double[][] costMatrix = deepCopy(originalCostMatrix);
		int m = costMatrix.length;
		int n = costMatrix[0].length;
		if (m != n) {
			costMatrix = padMatrix(costMatrix);
		}
		int[] assignments = new BipartiteMatcher().findBestAssignment(costMatrix);
		if (m != n) assignments = getPaddedMatching(assignments,m,n);
		double baseCost = getMatchingCost(originalCostMatrix, assignments);
		double[][] pathCosts = getPathResiduals(originalCostMatrix, assignments);
		for (int i=0; i<pathCosts.length; i++) {
			for (int j=0; j<pathCosts[i].length; j++) {
				pathCosts[i][j] += baseCost + originalCostMatrix[i][j];
			}
		}		
		return pathCosts;
	}
	
	public double getMatchingCost(double[][] costMatrix, int[] assignments) {
		double cost = 0;
		for (int i=0; i<assignments.length; i++) {			
			if (assignments[i] != -1)  cost += costMatrix[i][assignments[i]];						
		}
		return cost;
	}

	private double[][] getPathResiduals(double[][] originalCostMatrix, int[] assignments) {
		double[][] residualCostMatrix = getDirectedEdgeCostMatrix(originalCostMatrix, assignments);
		double[][] shortestPaths = new AllPairsShortestPath().getAllShortestPathCosts(residualCostMatrix);
		
		double[][] pathCosts = getUndirectedBipartiteGraphCostMatrix(shortestPaths, originalCostMatrix.length);
		return pathCosts;
	}

	private double[][] deepCopy(double[][] matrix) {
		if (matrix == null) {
			return null;
		}
		double[][] copy = new double[matrix.length][];
		for (int i=0; i<copy.length; i++) {
			copy[i] = new double[matrix[i].length];
			for (int j=0; j<copy[i].length; j++) {
				copy[i][j] = matrix[i][j];
			}
		}
		return copy;
	}

	private double[][] getDirectedEdgeCostMatrix(double[][] bipartiteCostMatrix, int[] assignments) {
		int n = bipartiteCostMatrix.length;
		int m = bipartiteCostMatrix[0].length;
		double[][] directedEdgeCostMatrix = new double[n+m][n+m];
		for (int i=0; i<n; i++) {
			for (int j=0; j<n; j++) {
				directedEdgeCostMatrix[i][j] = Double.POSITIVE_INFINITY;
			}
			for (int j=n; j<n+m; j++) {
				directedEdgeCostMatrix[i][j] = ( assignments[i] == (j-n) ) ? Double.POSITIVE_INFINITY : bipartiteCostMatrix[i][j-n];
			}
		}
		for (int i=n; i<n+m; i++) {
			for (int j=0; j<n; j++) {
				directedEdgeCostMatrix[i][j] = ( assignments[j] == (i-n) ) ? -bipartiteCostMatrix[j][i-n] : Double.POSITIVE_INFINITY;
			}
			for (int j=n; j<n+m; j++) {
				directedEdgeCostMatrix[i][j] = Double.POSITIVE_INFINITY;
			}
		}
		for (int i=0; i<n+m; i++) {
			directedEdgeCostMatrix[i][i] = 0;
		}
		return directedEdgeCostMatrix;
	}
	
	private double[][] getUndirectedBipartiteGraphCostMatrix(double[][] shortestPaths, int n) {
		int m = shortestPaths.length - n;
		double[][] bipartiteCostMatrix = new double[n][m];
		for (int i=0; i<n; i++) {
			for (int j=0; j<m; j++) {
				bipartiteCostMatrix[i][j] = shortestPaths[n+j][i];
			}
		}
		return bipartiteCostMatrix;
	}

}
