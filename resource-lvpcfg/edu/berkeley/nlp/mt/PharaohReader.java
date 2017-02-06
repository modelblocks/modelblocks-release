/**
 * 
 */
package edu.berkeley.nlp.mt;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import fig.basic.StrUtils;

/**
 * @author Alexandre Bouchard 
 * 
 * Edited by John DeNero to function with the
 *         transducer code base.
 */
public class PharaohReader implements AlignmentReader {
	public class AlignmentIterator implements Iterator<Alignment> {

		public boolean hasNext() {
			return PharaohReader.this.hasNext();
		}

		public Alignment next() {
			try {
				return PharaohReader.this.getNextAlignment();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (GizaFormatException e) {
				e.printStackTrace();
			}
			return null;
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	BufferedReader outSentences;
	BufferedReader inSentences;
	BufferedReader specsReader;
	boolean swap;
	
	public PharaohReader(BufferedReader outSentences, BufferedReader inSentences,
			BufferedReader specsReader, boolean swap) {
		this.outSentences = outSentences;
		this.inSentences = inSentences;
		this.specsReader = specsReader;
		this.swap = swap;
	}
	
	public PharaohReader(BufferedReader outSentences, BufferedReader inSentences,
			BufferedReader specsReader) {
		this(outSentences, inSentences, specsReader, false);
	}

	public Alignment getNextAlignment() throws IOException, GizaFormatException {
		List<String> out = StrUtils.splitByStr(outSentences.readLine().trim(), " ");
		List<String> in = StrUtils.splitByStr(inSentences.readLine().trim(), " ");
		Alignment a = new Alignment(out, in);
		String specs = specsReader.readLine().trim();
		a = parseAlignments(out, in, specs);
		return a;
	}

	public boolean hasNext() {
		try {
			return outSentences.ready() && inSentences.ready() && specsReader.ready();
		} catch (IOException e) {
			return false;
		}
	}

	public List<Alignment> getAllAlignments() throws IOException, GizaFormatException {
		List<Alignment> als = new ArrayList<Alignment>();
		while(hasNext()) {
			als.add(getNextAlignment());
		}
		return als;
	}

	public Iterator<Alignment> iterator() {
		return new AlignmentIterator();
	}

	/**
	 * Reads an alignment of the form
	 * 
	 * ([indexOfInWord]-[indexOfOutWord])*
	 * 
	 * e.g.
	 * 
	 * In : une phrase en francais 
	 * 
	 * Out: a french sentence
	 * 
	 * would be represented by:
	 * 
	 * "0-0 1-2 3-1"
	 * 
	 * If swap is set to true, the format is assumed to be flipped:
	 * 
	 * ([indexOfOutWord]-[indexOfInWord])*
	 * 
	 * If swap is not specified, it is assumed to be false.
	 * 
	 * @param outSentence
	 * @param inSentence
	 * @param specs
	 * @param swap
	 * @return
	 */
	public static Alignment parseAlignments(List<String> outSentence,
			List<String> inSentence, String specs, boolean swap, Alignment a) {
		Alignment result = a;
		for (String currentAlignmentPtSpec : StrUtils.splitByStr(specs, " ")) {
			if (currentAlignmentPtSpec.equals("")) {
				continue;
			}
			List<String> components = StrUtils.splitByStr(currentAlignmentPtSpec, "-");
			if (components.size() != 2) {
				throw new RuntimeException("Malformed alignment specifications.\nSpecs: " + specs
						+ "\nProblem is: " + currentAlignmentPtSpec);
			}
			try {
				int inPosition;
				int outPosition;
				if (swap) {
					inPosition = Integer.parseInt(components.get(1));
					outPosition = Integer.parseInt(components.get(0));
				} else {
					inPosition = Integer.parseInt(components.get(0));
					outPosition = Integer.parseInt(components.get(1));
				}
				result.addAlignment(outPosition, inPosition);
			} catch (NumberFormatException fne) {
				throw new RuntimeException("Malformed alignment specifications.\nSpecs: " + specs
						+ "\nProblem is: " + currentAlignmentPtSpec);
			}
		}
		return result;
	}
	
  
  public static Alignment parseAlignments(List<String> outSentence, List<String> inSentence, String specs, boolean swap)
  {
    return parseAlignments(outSentence, inSentence, specs, swap, new Alignment(outSentence, inSentence));
  }
  
  public static Alignment parseAlignments(List<String> outSentence, List<String> inSentence, String specs)
  {
    return parseAlignments(outSentence, inSentence, specs, false);
  }
  
  public static Alignment parseAlignments(String outSentence, String inSentence, String specs, boolean swap, Alignment a)
  {
    List<String> outSentenceList = StrUtils.splitByStr(outSentence, " ");
    List<String> inSentenceList = StrUtils.splitByStr(inSentence, " ");
    return parseAlignments(outSentenceList, inSentenceList, specs, swap, a);
  }
  
  public static Alignment parseAlignments(String outSentence, String inSentence, String specs, boolean swap)
  {
    List<String> outSentenceList = StrUtils.splitByStr(outSentence, " ");
    List<String> inSentenceList = StrUtils.splitByStr(inSentence, " ");
    return parseAlignments(outSentenceList, inSentenceList, specs, swap);
  }
  
  public static Alignment parseAlignments(String outSentence, String inSentence, String specs)
  {
    List<String> outSentenceList = StrUtils.splitByStr(outSentence, " ");
    List<String> inSentenceList = StrUtils.splitByStr(inSentence, " ");
    return parseAlignments(outSentenceList, inSentenceList, specs);
  }

}
