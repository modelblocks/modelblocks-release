package edu.berkeley.nlp.dep;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.util.CollectionUtils;
import fig.basic.Pair;

public class ViterbiDependencyParser {
	DependencyScorer depScorer;
	final int MAX_SENT_LENGTH = 200;

	// A span (s,t) is finished, means that
	// the head is in the interior of (s,t)
	// and not on the boundary (i.e not s or t)

	// Unfinished left scores
	double[][] ulScores;
	// Finished Left scores
	double[][] flScores;
	// Unfinished right scores
	double[][] urScores;
	// Finished right scores
	double[][] frScores;

	// Cached Dep Scores
	double[][] cachedDepScores;
	// Current Sentence length
	int curSentLength ;

	public ViterbiDependencyParser(DependencyScorer depScorer) {
		this.depScorer = depScorer;
		ulScores = new double[MAX_SENT_LENGTH][MAX_SENT_LENGTH];
		flScores = new double[MAX_SENT_LENGTH][MAX_SENT_LENGTH];
		urScores = new double[MAX_SENT_LENGTH][MAX_SENT_LENGTH];
		frScores = new double[MAX_SENT_LENGTH][MAX_SENT_LENGTH];
		cachedDepScores = new double[MAX_SENT_LENGTH][MAX_SENT_LENGTH];
	}

	private void clearArrays() {
		for (int s=0; s <= curSentLength; ++s) {
			Arrays.fill(ulScores[s], Double.NEGATIVE_INFINITY);
			Arrays.fill(flScores[s], Double.NEGATIVE_INFINITY);
			Arrays.fill(urScores[s], Double.NEGATIVE_INFINITY);
			Arrays.fill(frScores[s], Double.NEGATIVE_INFINITY);
		}
	}

	public void setInput(List<String> input) {
		if (!input.get(0).equals(DependencyConstants.BOUNDARY_WORD)) {
			throw new IllegalArgumentException("input doesn't start with $$");
		}
		this.curSentLength = input.size();
		clearArrays();
		cacheDepScores(input);
		initScores();
		insideProjection();
	}

	private void cacheDepScores(List<String> input) {
		depScorer.setInput(input);
		for (int s=0; s < input.size(); ++s) {
			for (int t=s+1; t < input.size(); ++t) {
				cachedDepScores[s][t+1] = Math.log(depScorer.getDependencyScore(s, t));
				cachedDepScores[t+1][s] = Math.log(depScorer.getDependencyScore(t, s));
			}
		}
	}

	private void insideProjection() {
		for (int len=2; len <= curSentLength; ++len) {
			for (int s=0; s+len <= curSentLength; ++s) {
				int t = s+len ;
				// Span [s,t)
				// Relevent dependencies are between s and (t-1)
				double leftDepScore = cachedDepScores[s][t];
				double rightDepScore = cachedDepScores[t][s];

				// Unfinished Left and Right
				// Each span can be of size 1
				for (int r=s+1; r < t; ++r) {
					double ulCur = flScores[s][r] + frScores[r][t] + leftDepScore;
					ulScores[s][t] = Math.max(ulScores[s][t], ulCur);
					double urCur = flScores[s][r] + frScores[r][t] + rightDepScore;
					urScores[s][t] = Math.max(urScores[s][t], urCur);
				}
				// Finished Left
				// Left Span has to be of length > 1
				// Spans can overlap
				for (int r=s+1; r < t; ++r) {
					assert (r+1) - s > 1;
					//  [s,r]-left-unfinished+ [r,t)-left-finished
					double flCur = ulScores[s][r+1] + flScores[r][t];
					flScores[s][t] = Math.max(flScores[s][t], flCur);
				}

				// Finished Right
				// Right span has to be of length > 1
				for (int r=s; r+1 < t; ++r) {
					assert t-r > 1;
					double frCur = frScores[s][r+1] + urScores[r][t];
					frScores[s][t] = Math.max(frScores[s][t], frCur);
				}
			}
		}
		if (flScores[0][curSentLength] == Double.NEGATIVE_INFINITY) {
			throw new IllegalStateException();
		}
	}

	private boolean approxEquals(double a, double b) {
		double diff = Math.abs(a-b);
		double min = Math.min(Math.abs(a),Math.abs(b));
		if (min == 0.0) {
			return diff < 1.0e-5;
		}
		return diff / min < 1.0e-5;
	}

	private Set<Pair<Integer, Integer>> decodeFinishedLeft(int s, int t) {
		assert s < t;
		if (t-s == 1) {
			return Collections.EMPTY_SET;
		}
		double goal = flScores[s][t];
		assert goal > Double.NEGATIVE_INFINITY;
		for (int r=s+1; r < t; ++r) {
			assert (r+1) - s > 1;
			//  [s,r]-left-unfinished+ [r,t)-left-finished
			double flCur = ulScores[s][r+1] + flScores[r][t];
			if (approxEquals(flCur,goal)) {
				Set<Pair<Integer, Integer>> result = new HashSet<Pair<Integer,Integer>>();
				result.addAll(  decodeUnfinishedLeft(s,r+1) );
				result.addAll(  decodeFinishedLeft(r,t) );
				return result;
			}
		}
		throw new IllegalStateException();
	}

	private Set<Pair<Integer, Integer>> decodeUnfinishedLeft(int s, int t) {
		assert s < t;
		double leftDepScore = cachedDepScores[s][t];
		double goal = ulScores[s][t];
		assert goal > Double.NEGATIVE_INFINITY;
		for (int r=s+1; r < t; ++r) {
			double ulCur = flScores[s][r] + frScores[r][t] + leftDepScore;
			if (approxEquals(ulScores[s][t], ulCur)) {
				Set<Pair<Integer, Integer>> result = new HashSet<Pair<Integer,Integer>>();
				result.add(Pair.newPair(s, t-1));
				result.addAll(decodeFinishedLeft(s,r));
				result.addAll(decodeFinishedRight(r, t));
				return result;
			}
		}
		throw new IllegalStateException();
	}

	private Set<Pair<Integer, Integer>> decodeUnfinishedRight(int s, int t) {
		assert s < t;
		double goal = urScores[s][t];
		assert goal > Double.NEGATIVE_INFINITY;
		double rightDepScore = cachedDepScores[t][s];
		for (int r=s+1; r < t; ++r) {
			double ulCur = flScores[s][r] + frScores[r][t] + rightDepScore;
			if (approxEquals(goal, ulCur)) {
				Set<Pair<Integer, Integer>> result = new HashSet<Pair<Integer,Integer>>();
				result.add(Pair.newPair(t-1, s));
				result.addAll(decodeFinishedLeft(s,r));
				result.addAll(decodeFinishedRight(r, t));
				return result;
			}
		}
		throw new IllegalStateException();
	}

	private Set<Pair<Integer, Integer>> decodeFinishedRight(int s, int t) {
		if (t-s == 1) {
			return Collections.EMPTY_SET;
		}
		double goal = frScores[s][t];
		assert goal > Double.NEGATIVE_INFINITY;
		for (int r=s; r+1 < t; ++r) {
			assert t-r > 1;
			double frCur = frScores[s][r+1] + urScores[r][t];
			if  (approxEquals(frCur, goal)) {
				Set<Pair<Integer, Integer>> result = new HashSet<Pair<Integer,Integer>>();
				result.addAll(decodeFinishedRight(s, r+1));
				result.addAll(decodeUnfinishedRight(r, t));
				return result;
			}
		}
		throw new IllegalStateException();
	}

	public Set<Pair<Integer,Integer>> decode() {
		try {
			return decodeFinishedLeft(0, curSentLength);
		} catch (Exception e) {
			System.err.println("Error Parisng");
			return new HashSet<Pair<Integer,Integer>>();
		}
	}

	private void initScores() {
		// Inititalize
		for (int s=0; s < curSentLength; ++s) {
			flScores[s][s+1] = frScores[s][s+1] = 0.0;
		}
	}

	public static void main(String[] args) {
		DependencyScorer depScorer = new DependencyScorer() {
			public double getDependencyScore(int head, int arg) {
				if (arg == 0) {
					return Double.NEGATIVE_INFINITY;
				}
				return Math.exp(-Math.abs(head-arg));
			}

			public void setInput(List<String> input) {
				// TODO Auto-generated method stub
			}
		};
		ViterbiDependencyParser viterbiParser = new ViterbiDependencyParser(depScorer);
		viterbiParser.setInput(CollectionUtils.makeList("$$","a","b","c","d","e"));
		Set<Pair<Integer, Integer>> result = viterbiParser.decode();
		System.out.println(result);
	}

}
