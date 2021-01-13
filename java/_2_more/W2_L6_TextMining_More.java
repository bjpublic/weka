package _2_more;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.trees.J48;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class W2_L6_TextMining_More extends W2_L4_TextMining_Base{
	 
	 public W2_L6_TextMining_More(String fileName) throws Exception {
		 super(fileName);
	 }
	 
	 public static void main(String[] args) throws Exception {
		  W2_L6_TextMining_More obj = new W2_L6_TextMining_More();
		  
		  new W2_L6_TextMining_More("ReutersGrain")
		     .suppliedTestSet(new J48());
		  new W2_L6_TextMining_More("ReutersGrain")
		     .suppliedTestSet(new NaiveBayesMultinomial());
		  new W2_L6_TextMining_More("ReutersGrain")
		     .reUseBaseEdition(new NaiveBayesMultinomial());
		  new W2_L6_TextMining_More("ReutersGrain")
		     .reUseBaseEdition(new J48());
	 }
	 
	 public void reUseBaseEdition(Classifier classifier) throws Exception{
		  // 1) FilteredClassifier °´Ã¼ È¹µæ
		  super.model = super.setModel(classifier);
		  
		  // 2) ÇÊÅÍ¼³Á¤
		  System.out.println("StringToWordVector ¼³Á¤°ª º¯°æ");
		  StringToWordVector word2vector = new StringToWordVector();
		  word2vector.setLowerCaseTokens(true);
		  word2vector.setOutputWordCounts(true);
		  word2vector.setStopwordsHandler(new Rainbow());
		  word2vector.setStemmer(new SnowballStemmer());
		  word2vector.setWordsToKeep(800);
		  super.model.setFilter(word2vector);
		  
		  // 3) ¸ðµ¨ÇÐ½À ½ÇÇà
		  super.suppliedTestSet(classifier);
	 }
	 
	 public W2_L6_TextMining_More(){
		 super();
	 }
}