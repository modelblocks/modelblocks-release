package edu.berkeley.nlp.util.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.tokenizer.PTBTokenizer;
import edu.berkeley.nlp.treebank.PennTreebankLanguagePack;
import edu.berkeley.nlp.treebank.TreebankLanguagePack;


public class DocumentSentenceSegmenter {

    public List<List<String>> getSentences(File file) {		
        StringBuilder data = new StringBuilder() ;
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            while (true) {
                String line = br.readLine();
                if (line == null) {
                    break;
                }
                data.append(line + "\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }		
        return getSentences(data.toString());		
    }

    private boolean stillInQuote(List<String> sent) {
        String openQuote = "``";
        String closeQuote = "''";
        int lastOpenIndex = sent.lastIndexOf(openQuote);
        int lastCloseQuote = sent.lastIndexOf(closeQuote);
        if (lastOpenIndex < 0) return false;
        if (lastCloseQuote < 0) return true;
        return lastCloseQuote <= lastOpenIndex;
    }


    public List<List<String>> getSentences(String docText) {		
        List<List<String>> sents = new ArrayList<List<String>>();
        List<String> curSent = new ArrayList<String>();
        TreebankLanguagePack langPack = new PennTreebankLanguagePack();
        List<String> puncToks = Arrays.asList(langPack.sentenceFinalPunctuationWords());
        try {
            PTBTokenizer toker = new PTBTokenizer(new StringReader(docText), false);
            List<String> allToks = toker.tokenize();
            for (String tok: allToks) {
                 
                curSent.add(tok);
                boolean isEnding = puncToks.contains(tok);
                boolean inQuote = stillInQuote(curSent);
                if (!inQuote && isEnding) {
                    sents.add(curSent);
                    curSent = new ArrayList<String>();
                }
                boolean isCloseQuote = tok.equals("''");
                boolean lastIsEnding =  curSent.size() > 1 && puncToks.contains(curSent.get(curSent.size()-2));
                if (isCloseQuote && lastIsEnding) {
                    sents.add(curSent);
                    curSent = new ArrayList<String>();
                }
            }
            if (!curSent.isEmpty()) {
                sents.add(curSent);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return sents;
    }

    public DocumentSentenceSegmenter() {

    }
    
    public static void main(String[] args) {
        String s = "`` But we have to attack the deficit . ''";
        List<List<String>> sents = new DocumentSentenceSegmenter().getSentences(s);
        for (List<String> sent : sents) {
            System.out.println(sent);
        }
    }


}
