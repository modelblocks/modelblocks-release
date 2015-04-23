package edu.berkeley.nlp.util.optionparser;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.annotation.ElementType;

/**
 * Created by IntelliJ IDEA.
* User: aria42
* Date: Oct 13, 2008
* Time: 5:11:32 PM
*/
@Retention(RetentionPolicy.RUNTIME)
@Target(value = {ElementType.FIELD,ElementType.METHOD})
public @interface Opt {
	public abstract String name() default "[unassigned]";
	public abstract String gloss() default "";
	public abstract boolean required() default false;
  public abstract String defaultVal() default "";
}
