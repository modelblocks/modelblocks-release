package edu.berkeley.nlp.concurrent;

import java.io.PrintWriter;
import java.util.Set;
import java.util.concurrent.Semaphore;

import edu.berkeley.nlp.util.LoggingWriter;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.LogInfo;
import fig.exec.Execution;

/**
 *  A reorderer that processes inputs in the order in which
 *  they were entered into a work queue.
 *  
 *  Note that this implementation uses existing threads to do all the work.
 */
public abstract class WorkQueueReorderer<T> {

	private PriorityQueue<T> pq = new PriorityQueue<T>();
	private Semaphore sem = new Semaphore(1);
	int nextToOutput = 0;
	private boolean dieOnException;

	
	public WorkQueueReorderer()
	{
		this(false);
	}
	public WorkQueueReorderer(boolean dieOnException)
	{
		this.dieOnException = dieOnException;
	}

	/**
	 * What to do with output, with order guarantees.
	 * 
	 * @param queueOutput Something created by a WorkQueue task
	 */
	public abstract void process(T queueOutput);

	public void addToProcessQueue(int orderIndex, T queueOutput) {
		sem.acquireUninterruptibly();

		if (orderIndex == nextToOutput) {
			nextToOutput++;
			try {
				process(queueOutput);
				drainQueue();
			}
			catch (Throwable e)
			{
				if (dieOnException)
				{
					Execution.raiseException(e);
					Execution.finish();
				}
				else
				{
					LogInfo.error("WorkQueueReorderer: " + e.getLocalizedMessage());
					e.printStackTrace(new PrintWriter(new LoggingWriter(true)));
				}
				
			}
		} else {
			pq.add(queueOutput, -1.0 * orderIndex);
		}
		sem.release();
	}

	private void drainQueue() {
		if (pq.isEmpty()) return;
		while (nextToOutput == -1.0 * pq.getPriority()) {
			process(pq.next());
			nextToOutput++;
			if(pq.isEmpty()) return;
		}
	}

	public boolean hasStrandedOutputs() {
		return !getStrandedOutputs().isEmpty();
	}

	public Set<T> getStrandedOutputs() {
		return pq.asCounter().keySet();
	}

}
