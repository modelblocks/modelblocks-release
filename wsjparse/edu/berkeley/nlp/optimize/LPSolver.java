package edu.berkeley.nlp.optimize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.MemoryUtils;
import fig.basic.Indexer;

public class LPSolver {

	private final Indexer varIndexer = new Indexer();
	private int[] varCounts  ;
	private PrintStream out;
	private String outFile ;
	private String inFile ;
	private String pathToSolver;
	private double maxSeconds = Double.POSITIVE_INFINITY;
	
	public LPSolver(String pathToSolver) {
		this.pathToSolver = pathToSolver;
	}

	private Counter getSolution(BufferedReader reader) {
		Counter varCounter = new Counter();
		boolean reachedStart = false;
		int numVars = 0;
		int numNonZeroVars = 0;
		while (true) {
			String line;			
			try {
				line = reader.readLine();
				if (line == null) {
					break;
				}
			
				if (reachedStart && line.equals("")) {
					break;
				}
				//System.out.println(line);
				if (line.contains("Column name")) {
					reachedStart = true;
					reader.readLine();
					continue;
				}
				if (reachedStart) {
					numVars++;
					String[] fields = line.trim().split("\\s+");
					int index = Integer.parseInt(fields[0])-1;
					double value = Double.parseDouble(fields[3]);
					if (value > 0.0) {
						numNonZeroVars++;
					}
					Object obj  = varIndexer.get(index);
					varCounter.setCount(obj, -value);					
				}
			} catch (IOException e) {				
				e.printStackTrace();
			}			
		}
		System.out.printf("Fraction of Non-Zero Vars: %.3f\n", numNonZeroVars / (float) numVars);
		
		return varCounter;
	}
	
	private int leftVarIndex(int index) {
		return index;
	}
	
	private void indexVariables(List<List>  items) {
		for  (List item: items) {
			for (Object elem: item) {
				varIndexer.add(elem);
			}
		}
	}

	private void countVariables(List<List> items) {
		varCounts = new int[varIndexer.size()];
		for (List item: items) {
			for (Object elem: item) {
				int varIndex = varIndexer.indexOf(elem);
				varCounts[varIndex]++;
			}			
		}
	}

	private void printHeader() {
		
	}

	private void printObjctive() {
		out.println("Maximize");
		out.print("value: ");
		for (int i=0; i < varCounts.length; ++i) {
			int count = varCounts[i];
			if (count == 0) {
				continue;
			}
			out.printf("%dx%d", count, leftVarIndex(i));			
			if (i+1 < varCounts.length) {
				out.printf(" +");			
			} 
			out.printf("\n");
		}
	}

	private void printConstraints(List<Double> values, List<List> items) {
		out.println("Subject To");
		for (int i=0; i < values.size(); ++i) {
			out.printf("c%d: ", i);
			List item = items.get(i);
			Counter constraintCounts  = new Counter();
			constraintCounts.incrementAll(item, 1.0);
			Iterator it = constraintCounts.keySet().iterator();
			while (it.hasNext()) {
				Object elem = it.next();
				int index = varIndexer.indexOf(elem);
				double count = constraintCounts.getCount(elem);
				out.printf("%dx%d",(int)count, index);
				if (it.hasNext()) {
					out.printf(" + ");
				}
			}
			
			out.printf(" <= %.3f\n", -values.get(i));
		}
	}

	private void printBounds() {
		out.println("Bounds");
		for (int i=0; i < varCounts.length; ++i) {
			out.printf("x%d >= 0\n", leftVarIndex(i));
		}
	}

	private void printTail() {
		out.println("End");
	}
	
	public void setMaxSeconds(double maxSeconds) {
		this.maxSeconds = maxSeconds;
	}

	private Counter solve(double[] values, List[] items) {
		return solve(doubleArrayAsList(values), Arrays.asList(items));
	}
	
	// List adapter for primitive double array
    private static List<Double> doubleArrayAsList(final double[] a) {
        return new AbstractList<Double>() {
            public Double get(int i) {
                return new Double(a[i]);
            }
            
            public int size() {
                return a.length;
            }
        };
    }
	
	public Counter solve(List<Double> values, List<List> items) {
		assert values.size() == items.size();
		System.out.println("In LP Solve: " + MemoryUtils.getHeapMemoryUsed());
		try {
			outFile = File.createTempFile("linprog", "prob").getAbsolutePath();
			inFile = File.createTempFile("linprog", "sol").getAbsolutePath();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		
		try {
			this.out =  new PrintStream(new File(outFile));
		} catch (Exception e) {
			e.printStackTrace();
		}

		indexVariables(items);
		countVariables(items);
		
		System.err.println("num constraints: " + items.size());
		System.out.println(outFile);
		
		printHeader();
		printObjctive();
		printConstraints(values, items);
		printBounds();
		printTail();
		
		try {			
			String[] cmd = {pathToSolver, "--cpxlp", "--tmlim", String.format("%d", (int) maxSeconds), "--nopresol", "-o",inFile , outFile};
			ProcessBuilder builder = new ProcessBuilder(cmd);
			builder.redirectErrorStream(true);
			System.out.println("Launching: " + builder.command());
			Process process = builder.start();			
			process.waitFor();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			while (true) {
				String line = reader.readLine();
				if (line == null) {
					break;
				}
				System.out.println(line);
			}
			
			System.out.println("process terminated");
			return getSolution(new BufferedReader(new FileReader(inFile)));
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		return null;
				
	}

	public static void main(String[] args) {
		double[] values = {Math.log(0.5),Math.log(0.5)};
		List p1 = CollectionUtils.makeList("a","a");
		List p2 = CollectionUtils.makeList("b","b");		
		List[] items = {p1,p2};
		String pathToSolver = "/usr/local/bin/glpsol";
		LPSolver lpSolver = new LPSolver(pathToSolver);
		System.out.println(lpSolver.solve(values, items));
	}


}
