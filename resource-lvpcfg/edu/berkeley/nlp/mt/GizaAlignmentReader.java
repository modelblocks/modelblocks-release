/**
 * 
 */
package edu.berkeley.nlp.mt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Utility class to read in a list of GIZA++ alignments from a file.
 * 
 * @param directory -
 *          the file to be read
 */
public class GizaAlignmentReader implements  AlignmentReader {

	public class GizaAlignmentIterator implements Iterator<Alignment> {

		public boolean hasNext() {
			return GizaAlignmentReader.this.hasNext();
		}

		public GizaAlignment next() {
			try {
				return GizaAlignmentReader.this.getNextAlignment();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (GizaFormatException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
		}

		public void remove() {
			throw new UnsupportedOperationException("Cannot remove alignments from a reader.");
		}

	}

	protected String fileName;
	protected BufferedReader fileReader;

	public GizaAlignmentReader(String fileName) {
		this.fileName = fileName;
		try {
			fileReader = new BufferedReader(new InputStreamReader(
					new FileInputStream(fileName), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public GizaAlignmentReader(BufferedReader reader) {
		fileReader = reader;
	}

	public GizaAlignment getNextAlignment() throws IOException,
			GizaFormatException {
		/*
		 * GIZA++ alignments come in sets of three lines. An example with added
		 * notation for new lines:
		 * 
		 * (line 1) # Sentence pair (1) source length 2 target length 2 alignment
		 * score : 0.0457472
		 * 
		 * (line 2) <CHAPTER ID=1>
		 * 
		 * (line 3) NULL ({ }) <CHAPTER ({ 1 }) ID=1> ({ 2 })
		 */
		String infoline = fileReader.readLine();
		String[] infowords = infoline.split("\\s+");
		if (infowords.length != 14)
			throw new GizaFormatException("Bad alignment file " + fileName
					+ ": wrong number of words in input line. Bad line was " + infoline);
		if (!infowords[0].equals("#"))
			throw new GizaFormatException("Bad alignment file " + fileName
					+ ": input line without initial #. Bad line was " + infoline);

		Integer sentenceID = Integer.parseInt(infowords[3].substring(1,
				infowords[3].length() - 1));
		Double score = Double.parseDouble(infowords[13]);

		String frenchline = fileReader.readLine();
		List<String> frenchWords = Arrays.asList(frenchline.split("\\s+"));

		String englishline = fileReader.readLine();
		List<String> englishWords = Arrays.asList(englishline.split("\\s+"));

		for (int i = 0; i < frenchWords.size(); i++) {
			frenchWords.set(i, frenchWords.get(i).intern());
		}

		for (int i = 0; i < englishWords.size(); i++) {
			englishWords.set(i, englishWords.get(i).intern());
		}

		return parseAlignment(frenchWords, englishWords, sentenceID, score,
				fileName);
	}

	public boolean hasNext() {
		try {
			return fileReader.ready();
		} catch (IOException e) {
			return false;
		}
	}

	public List<Alignment> getAllAlignments() throws IOException,
			GizaFormatException {
		List<Alignment> alignments = new ArrayList<Alignment>();
		while (hasNext()) {
			alignments.add(getNextAlignment());
		}
		return alignments;
	}

	/**
	 * Takes a GIZA++ alignment output file and returns a list of Alignments.
	 * 
	 * @param fileName
	 *          the input file name
	 * @return Returns a list of GizaAlignment objects.
	 */
	public static List<GizaAlignment> readAlignments(String fileName)
			throws IOException, GizaFormatException {
		List<GizaAlignment> alignments = new ArrayList<GizaAlignment>();
		BufferedReader in = new BufferedReader(new InputStreamReader(
				new FileInputStream(fileName), "UTF-8"));
		while (in.ready()) {
			/*
			 * GIZA++ alignments come in sets of three lines. An example with added
			 * notation for new lines:
			 * 
			 * (line 1) # Sentence pair (1) source length 2 target length 2 alignment
			 * score : 0.0457472
			 * 
			 * (line 2) <CHAPTER ID=1>
			 * 
			 * (line 3) NULL ({ }) <CHAPTER ({ 1 }) ID=1> ({ 2 })
			 */
			String infoline = in.readLine();
			String[] infowords = infoline.split("\\s+");
			if (infowords.length != 14)
				throw new GizaFormatException("Bad alignment file " + fileName
						+ ": wrong number of words in input line. Bad line was " + infoline);
			if (!infowords[0].equals("#"))
				throw new GizaFormatException("Bad alignment file " + fileName
						+ ": input line without initial #. Bad line was " + infoline);

			Integer sentenceID = Integer.parseInt(infowords[3].substring(1,
					infowords[3].length() - 1));
			Double score = Double.parseDouble(infowords[13]);

			String frenchline = in.readLine();
			List<String> frenchWords = Arrays.asList(frenchline.split("\\s+"));

			String englishline = in.readLine();
			List<String> englishLine = Arrays.asList(englishline.split("\\s+"));

			alignments.add(parseAlignment(frenchWords, englishLine, sentenceID,
					score, fileName));

		}
		return alignments;
	}

	/**
	 * Returns a GizaAlignment object from a pair of sentences where the english
	 * sentence is marked up according to the GIZA++ alignment output format.
	 * 
	 * @param frenchWords
	 *          the French sentence
	 * @param englishInput
	 *          the english line of the GIZA++ file (encoded)
	 * @param sentenceID
	 *          the GIZA++ sentence ID
	 * @param score
	 *          the GIZA++ score
	 * @param fileName
	 *          the GIZA++ filename
	 * @return Returns a GizaAlignment object.
	 */
	private static GizaAlignment parseAlignment(List<String> frenchWords,
			List<String> englishInput, Integer sentenceID, Double score,
			String fileName) throws GizaFormatException {
		List<String> englishWords = new ArrayList<String>();
		int alignmentsFromFrenchToEnglish[] = new int[frenchWords.size() + 1];
		// First, we extract the alignments and english words from the english.
		// Format: NULL ({ }) <CHAPTER ({ 1 }) ID=1> ({ 2 })
		int englishPosition = 0;
		int inputPosition = 0;
		while (inputPosition < englishInput.size()) {
			if (englishPosition != 0) { // Not the NULL word
				englishWords.add(englishInput.get(inputPosition));
				inputPosition++;
				if (!"({".equals(englishInput.get(inputPosition)))
					throw new GizaFormatException(
							"Improperly formed english input string at sentence #"
									+ sentenceID);
				inputPosition++;
				while (!"})".equals(englishInput.get(inputPosition))) {
					try {
						int french = Integer.parseInt(englishInput.get(inputPosition));
						alignmentsFromFrenchToEnglish[french] = englishPosition;
						inputPosition++;
					} catch (NumberFormatException nfe) {
						throw new GizaFormatException(
								"Improperly formed english input string at sentence #"
										+ sentenceID);
					}
				}
			} else { // Skip past NULL information
				while (!"})".equals(englishInput.get(inputPosition))) {
					inputPosition++;
				}
			}
			inputPosition++;
			englishPosition++;
		}

		// Then, we build an alignment structure.
		GizaAlignment alignment = new GizaAlignment(englishWords, frenchWords,
				score, sentenceID, fileName);
		for (int frenchPosition = 1; frenchPosition <= frenchWords.size(); frenchPosition++) {
			// Here, we account for the fact that GIZA++ output is
			// 1-indexed while our alignment class is 0-indexed.
			alignment.addAlignment(alignmentsFromFrenchToEnglish[frenchPosition] - 1,
					frenchPosition - 1);
		}

		return alignment;
	}

	public Iterator<Alignment> iterator() {
		return new GizaAlignmentIterator();
	}
}
