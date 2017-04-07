package edu.berkeley.nlp.mt;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public interface AlignmentReader extends Iterable<Alignment>{

	public abstract Alignment getNextAlignment() throws IOException,
			GizaFormatException;

	public abstract boolean hasNext();

	public abstract List<Alignment> getAllAlignments() throws IOException,
			GizaFormatException;

	public abstract Iterator<Alignment> iterator();

}