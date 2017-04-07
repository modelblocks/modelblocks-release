package edu.berkeley.nlp.util.optionparser;

/**
 * Created by IntelliJ IDEA.
 * User: aria42
 * Date: Oct 15, 2008
 * Time: 12:26:24 AM
 */
public abstract class GlobalOptionsClass {
		public GlobalOptionsClass() {
		 	GlobalOptionParser.fillOptions(this);
		}
}
