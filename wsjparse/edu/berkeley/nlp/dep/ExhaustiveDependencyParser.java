package edu.berkeley.nlp.dep;

import java.util.Arrays;
import java.util.List;
import edu.berkeley.nlp.util.CollectionUtils;

public class ExhaustiveDependencyParser {

	// Unfinished left scores
	double[][] ulInsScores; int[][] ulInsScales;
	double[][] ulOutScores; int[][] ulOutScales;
	// Finished Left scores
	double[][] flInsScores; int[][] flInsScales;
	double[][] flOutScores; int[][] flOutScales;
	// Unfinished right scores
	double[][] urInsScores; int[][] urInsScales;
	double[][] urOutScores; int[][] urOutScales;
	// Finished right scores
	double[][] frInsScores; int[][] frInsScales;
	double[][] frOutScores; int[][] frOutScales;

	// Scalers
	Scaler ulScaler = new Scaler();
	Scaler urScaler = new Scaler();
	Scaler flScaler = new Scaler();
	Scaler frScaler = new Scaler();

	// Dep Scorer
	DependencyScorer depScorer;

	// Cached Dep Scores
	double[][] cachedDepScores;
	// Dep Posteriors
	double[][] depPosteriors ;
	// Current Sentence length
	int curSentLength ;
	// Max Sent Length
	private static final int MAX_SENT_LEN = 200;

	public ExhaustiveDependencyParser(DependencyScorer depScorer) {
		this.depScorer = depScorer;
		createArrays();
	}

	private void createArrays() {
		ulInsScores = newDoubleArray();
		urInsScores = newDoubleArray();
		ulOutScores = newDoubleArray();
		urOutScores = newDoubleArray();
		flInsScores = newDoubleArray();
		frInsScores = newDoubleArray();
		flOutScores = newDoubleArray();
		frOutScores = newDoubleArray();

		ulInsScales = newIntArray();
		urInsScales = newIntArray();
		ulOutScales = newIntArray();
		urOutScales = newIntArray();
		flInsScales = newIntArray();
		frInsScales = newIntArray();
		flOutScales = newIntArray();
		frOutScales = newIntArray();

		cachedDepScores = newDoubleArray();
		depPosteriors = newDoubleArray();
	}

	private double[][] newDoubleArray() {
		return  new double[MAX_SENT_LEN][MAX_SENT_LEN];
	}

	private int[][] newIntArray() {
		return new int[MAX_SENT_LEN][MAX_SENT_LEN];
	}

	private String toString(double[][] matrix) {
		StringBuilder sb = new StringBuilder();
		for (int i=0; i < curSentLength; ++i) {
			double[] row = new double[curSentLength+1];
			System.arraycopy(matrix[i], 0, row, 0, curSentLength+1);
			sb.append(Arrays.toString(row));
			sb.append("\n");
		}
		return sb.toString();
	}

	public void setInput(List<String> input) {
		if (!input.get(0).equals(DependencyConstants.BOUNDARY_WORD)) {
			throw new IllegalArgumentException();
		}
		this.curSentLength = input.size();
		clearArrays();
		cacheDepScores(input);
		insideProjections();
		outsideProjections();
		computeDepPosteriors();
	}

	private void computeDepPosteriors() {
		double z = flInsScores[0][curSentLength];
		int z_scale = flInsScales[0][curSentLength];
		for (int s=0; s < curSentLength; ++s) {
			for (int t=s+1; t < curSentLength; ++t) {
				depPosteriors[s][t] = depPosteriors[t][s] = 0.0;

				double left_posterior_unscaled = ulOutScores[s][t+1] *  ulInsScores[s][t+1] / z;
				int left_posterior_scale = ulOutScales[s][t+1] + ulInsScales[s][t+1] - z_scale;
				depPosteriors[s][t] += ulScaler.getScaled(left_posterior_unscaled, left_posterior_scale);

				double right_posterior_unscaled = urOutScores[s][t+1] * urInsScores[s][t+1] / z;
				int right_posterior_scale = urOutScales[s][t+1] + urInsScales[s][t+1] - z_scale;
				depPosteriors[t][s] += urScaler.getScaled(right_posterior_unscaled, right_posterior_scale);
			}
		}
	}

	private void cacheDepScores(List<String> input) {
		depScorer.setInput(input);
		for (int s=0; s < input.size(); ++s) {
			for (int t=s+1; t < input.size(); ++t) {
				cachedDepScores[s][t+1] = depScorer.getDependencyScore(s, t);
				cachedDepScores[t+1][s] = depScorer.getDependencyScore(t, s);
			}
		}
	}

	private void clearArrays() {
		for (int s=0; s < curSentLength; ++s) {
			for (int t=s+1; t <= curSentLength; ++t) {
				ulInsScores[s][t] = 0.0;  ulOutScores[s][t] = 0.0;
				urInsScores[s][t] = 0.0; urOutScores[s][t] = 0.0;
				flInsScores[s][t] = 0.0;  flOutScores[s][t] = 0.0;
				frInsScores[s][t] = 0.0;  frOutScores[s][t] = 0.0;

				ulInsScales[s][t] = 0;  ulOutScales[s][t] = 0;
				urInsScales[s][t] = 0; urOutScales[s][t] = 0;
				flInsScales[s][t] = 0;  flOutScales[s][t] = 0;
				frInsScales[s][t] = 0; frOutScales[s][t] = 0;
			}
		}
	}

	private void insideProjections(	) {
		// Init
		for (int s=0; s < curSentLength; ++s) {
			flInsScores[s][s+1] = 1.0;
			frInsScores[s][s+1] = 1.0;
		}

		for (int len=2; len <  curSentLength; ++len) {
			for (int s=1; s+len <= curSentLength; ++s) {
				int t = s + len;

				// Clear Scale Summers
				ulScaler.clear(); urScaler.clear();
				flScaler.clear(); frScaler.clear();

				// Span [s,t)
				// Relevent dependencies are between s and (t-1)
				double leftDepScore = cachedDepScores[s][t];
				double rightDepScore = cachedDepScores[t][s];

				// Unfinished Left and Right
				// Each span can be of size 1
				for (int r=s+1; r < t; ++r) {
					double ulScore = flInsScores[s][r] * frInsScores[r][t] * leftDepScore;
					int ulScale = flInsScales[s][r] + frInsScales[r][t];
					ulScaler.add(ulScore, ulScale);
					double urScore = flInsScores[s][r] * frInsScores[r][t] * rightDepScore;
					int urScale = flInsScales[s][r] + frInsScales[r][t];
					urScaler.add(urScore, urScale);
				}
				// Scale and Get Sum
				ulScaler.scale(); urScaler.scale();
				ulInsScores[s][t] = ulScaler.getSumUnscaled(); ulInsScales[s][t] = ulScaler.getSumScale();
				urInsScores[s][t] = urScaler.getSumUnscaled(); urInsScales[s][t] = urScaler.getSumScale();

				// Finished Left
				// Left Span has to be of length > 1
				// Spans can overlap
				for (int r=s+1; r < t; ++r) {
					assert (r+1) - s > 1;
					double flUnscaled = ulInsScores[s][r+1] * flInsScores[r][t];
					int flScale = ulInsScales[s][r+1] +  flInsScales[r][t];
					flScaler.add(flUnscaled, flScale);
				}
				flScaler.scale();
				flInsScores[s][t] = flScaler.getSumUnscaled();
				flInsScales[s][t] = flScaler.getSumScale();

				// Finished Right
				// Right span has to be of length > 1
				// s=1, t=5
				for (int r=s; r+1 < t; ++r) {
					assert t - r > 1;
					double frUnscaled = frInsScores[s][r+1] * urInsScores[r][t];
					int frScale = frInsScales[s][r+1] + urInsScales[r][t];
					frScaler.add(frUnscaled, frScale);
				}
				frScaler.scale();
				frInsScores[s][t] = frScaler.getSumUnscaled();
				frInsScales[s][t] = frScaler.getSumScale();
			}
		}

		// Length Case
		flScaler.clear();
		for (int s=1; s < curSentLength; ++s) {
			// What if s is the root?
			ulInsScores[0][s+1] = flInsScores[0][1] * frInsScores[1][s+1] * depScorer.getDependencyScore(0, s);
			ulInsScales[0][s+1] =  flInsScales[0][1] + frInsScales[1][s+1];
			double flScore = ulInsScores[0][s+1] * flInsScores[s][curSentLength];
			int flScale = ulInsScales[0][s+1] + flInsScales[s][curSentLength];
			flScaler.add(flScore, flScale);
		}
		flScaler.scale();
		flInsScores[0][curSentLength] = flScaler.getSumUnscaled();
		flInsScales[0][curSentLength] =  flScaler.getSumScale();


		double z = flInsScores[0][curSentLength];
		int z_scale = flInsScales[0][curSentLength];
		double logZ = Math.log(z) + z_scale * ulScaler.getLogScale();
		if (logZ == Double.NEGATIVE_INFINITY) {
			boolean ignore = true;
		}
		System.out.printf("logZ: %.5f\n",logZ);

	}

	private void outsideProjections() {
		// Init
		flOutScores[0][curSentLength] = 1.0;
		ulOutScores[0][curSentLength] = 1.0;

		for (int len=curSentLength-1; len > 0; --len) {
			for (int s=0; s+len <= curSentLength; ++s) {
				int r = s + len;

				// Clear Scale Summers
				ulScaler.clear();  urScaler.clear();
				flScaler.clear();   frScaler.clear();

				// Finished Left
				// Two Cases:
				// (1) To Create Unfinished Span
				if (flInsScores[s][r] > 0.0) {
					for (int t=r+1; t <= curSentLength; ++t) {
						// UL(s,t) = FL(s,r) + FR(r,t) + dep(s,t)
						double flCur = ulOutScores[s][t] * frInsScores[r][t] * cachedDepScores[s][t];
						int flScale = ulOutScales[s][t] + frInsScales[r][t];
						flScaler.add(flCur, flScale);
						// UR(s,t) = FL(s,r) + FR(r,t) + dep(t,s)
						flCur = ulOutScores[s][t] * frInsScores[r][t] * cachedDepScores[t][s];
						flScale = ulOutScales[s][t] + frInsScales[r][t];
						flScaler.add(flCur, flScale);
					}
					// (2) To Create Finished Span
					for (int a=0; a < s; ++a) {
						// FL(a,r) = UL(a,s+1) + FL(s,r)
						double flScore = flOutScores[a][r] * ulInsScores[a][s+1] ;
						int flScale =  flOutScales[a][r]  +ulInsScales[a][s+1];
						flScaler.add(flScore, flScale);
					}
					flScaler.scale();
					flOutScores[s][r] = flScaler.getSumUnscaled();
					flOutScales[s][r] = flScaler.getSumScale();
				}
				// Done Finished Left

				// Finished Right
				// Two Cases:
				// (1) To Create Unfinished Span
				if (frInsScores[s][r] > 0.0) {
					for (int a=0; a < s; ++a) {
						// UL(a,r) = FL(a,s) + FR(s,r)
						double frScore = ulOutScores[a][r] * flInsScores[a][s] * cachedDepScores[a][r];
						int frScale = ulOutScales[a][r] + flInsScales[a][s] ;
						frScaler.add(frScore, frScale);
						//	UR(a,r) = FL(a,s) + FR(s,r)
						frScore = urOutScores[a][r] * flInsScores[a][s] * cachedDepScores[r][a];
						frScale = urOutScales[a][r] + flInsScales[a][s] ;
						frScaler.add(frScore, frScale);
					}
					// (2) To Create Finished Span
					if (r > 0) {
						for (int t=r+1; t <= curSentLength; ++t) {
							// FR(s,t) = FR(s,r) UR(r-1,t)
							double frScore = frOutScores[s][t] * urInsScores[r-1][t];
							int frScale = frOutScales[s][t] + urInsScales[r-1][t];
							frScaler.add(frScore, frScale);
						}
					}
					frScaler.scale();
					frOutScores[s][r] = frScaler.getSumUnscaled();
					frOutScales[s][r] = frScaler.getSumScale();
				}
				// Done Finished Right


				// Unfinished Spans //
				if (r-s >1) {
					// Unfinished Left
					// Expand the Finished Left Scores to the Right
					// FL(s,t) = UL(s,r) + FL(r-1, t)
					if (ulInsScores[s][r] > 0.0) {
						for (int t=r; r > 0 && t <= curSentLength; ++t) {
							double ulScore = flOutScores[s][t] * flInsScores[r-1][t] ;
							int ulScale = flOutScales[s][t] + flInsScales[r-1][t];
							ulScaler.add(ulScore, ulScale);
						}
						ulScaler.scale();
						ulOutScores[s][r] = ulScaler.getSumUnscaled();
						ulOutScales[s][r] = ulScaler.getSumScale();
					}
					//	Unfinished right
					// Expand the Finished Right Scores to the Left
					// FR(a,r) = FR(a,s+1) + UR(s,r)   // (r-s > 1)
					if (urInsScores[s][r] > 0.0) {
						for (int a=0; a < s+1; ++a) {
							double urCur = frOutScores[a][r] * frInsScores[a][s+1];
							int urScale = frOutScales[a][r] + frInsScales[a][s+1];
							urScaler.add(urCur, urScale);
						}
						urScaler.scale();
						urOutScores[s][r] = urScaler.getSumUnscaled();
						urOutScales[s][r] = urScaler.getSumScale();
					}
				}

			}
		}
	}

	public double[][] getDependencyPosteriors() {
		return depPosteriors;
	}

	public static void main(String[] args) {
		DependencyScorer depScorer = new DependencyScorer() {
			public double getDependencyScore(int head, int arg) {
				if (arg == 0) {
					return 0.0;
				}

				return 1.0;
			}

			public void setInput(List<String> input) {
				// TODO Auto-generated method stub

			}
		};
		ExhaustiveDependencyParser depParser = new ExhaustiveDependencyParser(depScorer);
		List<String> input = CollectionUtils.makeList("$$","a","b","c","d");
		depParser.setInput(input);
		System.out.println(depParser.toString(depParser.depPosteriors));
	}
}
