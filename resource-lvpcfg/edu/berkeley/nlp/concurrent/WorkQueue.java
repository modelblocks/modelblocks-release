package edu.berkeley.nlp.concurrent;

import java.io.PrintWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import edu.berkeley.nlp.util.LoggingWriter;
import fig.basic.LogInfo;
import fig.exec.Execution;

/**
 * A thread manager for executing many tasks safely using a fixed number of
 * threads.
 * 
 * Use WorkQueueReorderer to recover ordered outputs
 */
public class WorkQueue {

	private static final long WAIT_TIME = 10;

	private ExecutorService executor;

	private Semaphore sem;

	private boolean serialExecution;

	private boolean dieOnException;

	public WorkQueue(int numThreads) {
		this(numThreads, false);
	}

	public WorkQueue(int numThreads, boolean dieOnException) {
		this.dieOnException = dieOnException;
		if (numThreads == 0) {
			serialExecution = true;
		} else {
			executor = Executors.newFixedThreadPool(numThreads);
			sem = new Semaphore(numThreads);
			serialExecution = false;
		}
	}

	public void execute(final Runnable work) {
		if (serialExecution) {
			work.run();
		} else {
			sem.acquireUninterruptibly();
			executor.execute(new Runnable() {

				public void run() {
					if (!dieOnException) {
						try {
							work.run();
						} catch (AssertionError e) {

							LogInfo.error(e);
							e.printStackTrace(new PrintWriter(
									new LoggingWriter(true)));
						} catch (RuntimeException e) {
							LogInfo.error(e);

							e.printStackTrace(new PrintWriter(
									new LoggingWriter(true)));
						}
					} else {
						try {
							work.run();
						} catch (Throwable t) {
							Execution.raiseException(t);
							Execution.finish();
						}

					}
					sem.release();
				}
			});
		}
	}

	public void finishWork() {
		if (serialExecution) return;
		executor.shutdown();
		try {
			int secs = 0;
			while (!executor.awaitTermination(WAIT_TIME, TimeUnit.SECONDS)) {
				secs += WAIT_TIME;
				LogInfo.logs("Awaited executor termination for %d seconds",
						secs);
			}
		} catch (InterruptedException e) {
			throw new RuntimeException("Work queue interrupted");
		}
	}
}
