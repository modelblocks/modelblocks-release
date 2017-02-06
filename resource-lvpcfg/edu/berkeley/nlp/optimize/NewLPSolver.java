package edu.berkeley.nlp.optimize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

import edu.berkeley.nlp.util.Counter;
import fig.basic.Indexer;

public class NewLPSolver {

	String constFile;
	String objFile ;
	String solFile;
	double maxSeconds = Double.POSITIVE_INFINITY;
	final String pathToSolver = System.getenv("GLPSOL");
	int numConstraints = 0;
	
	PrintStream constOut, objOut;

	private boolean firstObjVar = true;
	private Indexer indexer = new Indexer();
	
	public void setMaxSeconds(double maxSeconds) {
		this.maxSeconds = maxSeconds;
	}
	
	public NewLPSolver(boolean maximize) {


		try {
			
			constFile = File.createTempFile("linearprog", "const").getAbsolutePath();
			objFile = File.createTempFile("linearprog", "obj").getAbsolutePath();
			solFile = File.createTempFile("linearprog", "sol").getAbsolutePath();
			constOut = new PrintStream(constFile);
			objOut = new PrintStream(objFile);
			
		} catch (IOException e) {
			e.printStackTrace();
		}

		if (maximize) {
			objOut.println("Maximize");
		} else {
			objOut.println("Minimize");
		}
		objOut.print("value: ");
		
		constOut.println("Subject to");
	}
	
	public void addObjectiveTerm(Object var, double val) {
		
		indexer.add(var);
		int index = indexer.indexOf(var);
		
		if (firstObjVar) {
			objOut.printf("%.5fx%d\n",val,index);
			firstObjVar = false;
		} else {
			objOut.printf("+ %.5fx%d\n",val,index);
		}
		
	}
	
	public void addLessThanConstraint(double val,Object...vars) {
		for (int i=0; i < vars.length; ++i) {
			Object var = vars[i];
			indexer.add(var);
			int index = indexer.indexOf(var);			
			constOut.printf("x%d ",index);
			if (i+1 < vars.length) {
				constOut.printf("+ ");
			}
		}				
		constOut.printf(" <= %.5f\n",val);
		numConstraints++;
	}
	
	private Counter getSolution(BufferedReader reader) {
		Counter solution = new Counter();
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
					int index = Integer.parseInt(fields[1].substring(1));
					double value = Double.parseDouble(fields[3]);
					if (value > 0.0) {
						numNonZeroVars++;
					}
					Object var = indexer.get(index);
					solution.setCount(var, value);
										
				}
			} catch (IOException e) {				
				e.printStackTrace();
			}			
		}
		System.out.printf("Fraction of Non-Zero Vars: %.3f\n", numNonZeroVars / (float) numVars);
		
		return solution;
	}
	
	
	public void addGreaterThanConstraint(double val, Object...vars) {
		for (int i=0; i < vars.length; ++i) {
			Object var = vars[i];
			indexer.add(var);
			int index = indexer.indexOf(var);			
			constOut.printf("x%d ",index);		
			if (i+1 < vars.length) {
				constOut.printf("+ ");
			}
		}
		constOut.printf(" >= %.5f\n",val);
		numConstraints++;
	}
	
	public void writeProblem(String file)  {
				
		PrintStream out = null;
		try {			
			out =  new PrintStream(file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		copy(objFile, out);
		copy(constFile, out);
		out.println("end");
	}
	
	public Counter solve() {
		try {
			String probFile = File.createTempFile("linearprog", "prob").getAbsolutePath();
			writeProblem(probFile);
			
			System.err.println("Num Variavles: " + indexer.size());
			System.err.println("Num Constraints: " + numConstraints);
			
			String solFile = File.createTempFile("linearprog", "sol").getAbsolutePath();
			String[] cmd = {pathToSolver, "--cpxlp",  "--tmlim", String.format("%d", (int) maxSeconds), "--nopresol", "-o",solFile, probFile};
			ProcessBuilder builder = new ProcessBuilder(cmd);
			builder.redirectErrorStream(true);
			System.out.println("Launching: " + builder.command());
			Process process = builder.start();			
			//process.waitFor();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			while (true) {
				String line = reader.readLine();
				if (line == null) {
					break;
				}
				System.out.println(line);
			}
			System.err.println("Process terminated");
			return getSolution(new BufferedReader(new FileReader(solFile)));
			
		} catch (Exception e) {			
			e.printStackTrace();
			System.exit(0);
		}
		return null;
	}
	
	private void copy(String file, PrintStream out) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			try {
				while (true) {
					String line = br.readLine();
					if (line == null) {
						break;
					}
					out.println(line);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		NewLPSolver lpSolver = new NewLPSolver(true);
		lpSolver.addObjectiveTerm(0, 1.0);
		lpSolver.addObjectiveTerm(1, 1.0);
		lpSolver.addLessThanConstraint(1.0,0,1);
		System.out.println(lpSolver.solve());
	}
}
