package edu.berkeley.nlp.util.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import edu.berkeley.nlp.tokenizer.PTBLineLexer;
import edu.berkeley.nlp.util.ConcatenationIterator;
import edu.berkeley.nlp.util.Filter;
import edu.berkeley.nlp.util.IOUtil;
import edu.berkeley.nlp.util.Iterators;
import edu.berkeley.nlp.util.Method;
import edu.berkeley.nlp.util.StringUtils;
import fig.basic.IOUtils;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

/**
 * Reusable iterator over the sentences in a document collection. The class is careful to not load
 * all the data into memory but instead will only queue a small number in memory. You can pass options
 * to do things like tokenize and case fold.
 *
 * @author aria42
 *
 */
public class SentenceIterable implements Iterable<List<String>>, Runnable {

	@Option(required=true)
	public String dataRoot ;
	@Option(gloss="Prefix of files to consider")
	public String prefix = "";
	@Option(gloss="Extension of files to consider. If you pass .gz as a suffix, we will unzip them with gzip")
	public String extension = ".txt";
	@Option(gloss="Maxixum number of sentences")
	public int maxNumSentences = Integer.MAX_VALUE;
	@Option(gloss="Tokenize each sentence")
	public boolean tokenize = false;
	@Option(gloss="Lowercase data")
	public boolean lowercase = false;
	@Option(gloss="How many sentences to buffer")
	public int bufferSize = 1;
	@Option(gloss="Maximum number of sentences")
	public int maxSentenceLength = Integer.MAX_VALUE;
	@Option(gloss="Do we need to segment senteces first. Forces tokenize=true")
	public boolean sentenceSegment = false;

	private Iterable<File> files ;

	public SentenceIterable(List<File> files) {
		this.files = files;
	}

	public SentenceIterable() {

	}

	public void setTokenize(boolean tokenize) {
		this.tokenize = tokenize;
	}

	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}

	public void setExtension(String extension) {
		this.extension = extension;
	}

	public boolean isLowercase() {
		return lowercase;
	}

	public void setLowercase(boolean lowercase) {
		this.lowercase = lowercase;
	}

	public class MyIterator implements Iterator<List<String>> {

		Iterator<File> fileIt ;
		Iterator<List<String>> curLinesIt = Iterators.emptyIterator();

		public MyIterator(Iterator<File> fileIt) {
			this.fileIt = fileIt;
		}

		public boolean hasNext() {
			// TODO Auto-generated method stub
			return queueNext();
		}

		public List<String> next() {
			boolean hasNext = queueNext();
			if (!hasNext) {
				throw new IllegalStateException();
			}
			return curLinesIt.next();
		}

		private boolean queueNext() {
			if (curLinesIt.hasNext()) {
				return true;
			}
			if (!fileIt.hasNext()) {
				return false;
			}
			curLinesIt = nextFileLineIterator();
			return queueNext();
		}

		private Iterator<List<String>> nextFileLineIterator()  {
			DocumentSentenceSegmenter sentSegmenter = new DocumentSentenceSegmenter();
			File file = fileIt.next();
			Iterator<List<String>> sentIter = null;
			if (sentenceSegment) {
				sentIter = sentSegmenter.getSentences(file).iterator();
			} else {
				try {
					Iterator<String> lineIt = IOUtils.lineIterator(file.getAbsolutePath());
					final PTBLineLexer toker = new PTBLineLexer();
					Method<String, List<String>> m = new Method<String, List<String>>() {
						public List<String> call(String obj) {
							List<String> toks = null;
							if (tokenize) {
								try {
									toks = toker.tokenizeLine(obj);
								} catch (IOException e) {
									e.printStackTrace();
									System.exit(0);
								}
							} else {
								toks = Arrays.asList(obj.split("\\s+"));
							}
							return toks;
						}
					};
					sentIter = new Iterators.TransformingIterator<String, List<String>>(lineIt, m);
				} catch (IOException e) {
					e.printStackTrace();
					System.exit(0);
				}
			}
			if (lowercase) {
				sentIter = new Iterators.TransformingIterator<List<String>, List<String>>(sentIter,
							new Method<List<String>, List<String>>() {
							public List<String> call(List<String> obj) {
								for (int i=0; i < obj.size(); ++i) {
									obj.set(i, obj.get(i).toLowerCase());
								}
								return obj;
							}
				});
			}
			return sentIter;
		}

		public void remove() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException();
		}



	}
	
	public Iterator<List<String>> iterator() {
		List<File> files = IOUtils.getFilesUnder(dataRoot, IOUtil.getFileFilter(prefix, extension));
		Iterator<List<String>> it = new MyIterator(files.iterator());
		Filter<List<String>> filter = new Filter<List<String>>() { public boolean accept(List<String> t) { return t.size() <= maxSentenceLength; }};
		it = Iterators.filter(it,filter);
		return Iterators.maxLengthIterator(it, maxNumSentences);
	}


	public static class MainRunOptions {
		@Option(required=true,gloss="Where to put data one-sentence per-line")
		public String outDir ;
		@Option(gloss="Extension of output data")
		public String outExtension = ".tok";
	}

	public void run() {
		List<File> files = IOUtils.getFilesUnder(dataRoot, IOUtil.getFileFilter(prefix, extension));
		IOUtils.createNewDirIfNotExistsEasy(mainOpts.outDir);
		for (File file: files) {
			Iterator<List<String>> it = new MyIterator(Collections.singletonList(file).iterator());
			Filter<List<String>> filter = new Filter<List<String>>() { public boolean accept(List<String> t) { return t.size() <= maxSentenceLength; }};
			it = Iterators.filter(it,filter);
			it = Iterators.maxLengthIterator(it, maxNumSentences);
			List<String> lines = Iterators.fillList(
					new Iterators.TransformingIterator<List<String>, String>(it,
							new Method<List<String>, String>() {
								public String call(List<String> input) {
									return StringUtils.join(input, " ");
								}
							}));
			String outfile = file.getName() + mainOpts.outExtension;
			File outpath = new File(mainOpts.outDir,outfile);
			IOUtils.writeLinesHard(outpath.getAbsolutePath(), lines);
		}
	}

	private static MainRunOptions mainOpts = new MainRunOptions();

	public static void main(String[] args) {
		SentenceIterable sentIterable = new SentenceIterable();
		Execution.run(args,sentIterable, mainOpts);
	}
}
