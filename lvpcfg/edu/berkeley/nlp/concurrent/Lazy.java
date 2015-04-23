package edu.berkeley.nlp.concurrent;

/**
 *	A lazy computation instance that yields a T.
 *
 *  This can be used to encapsulate arbitrary parameters 
 *  and inputs for future processing.
 *  
 *  Be sure that get can be called concurrently on many
 *  lazy objects
 */
public interface Lazy<T> {
	public T get();
}
