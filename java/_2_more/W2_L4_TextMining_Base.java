package _2_more;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class W2_L4_TextMining_Base {

	 Instances train=null;
	 Instances test=null;
	
	 FilteredClassifier model = null;
	
	 // 생성자
	 public W2_L4_TextMining_Base(String fileName) throws Exception{
		 
		  // 1) data loader
		  String folderName = "D:\\Weka-3-9\\data\\";
		  System.out.print(fileName + " ");
		  this.train=new Instances(
		            new BufferedReader(
		            new FileReader(folderName+fileName+"-train-edited.arff")));
		  this.test =new Instances(
		            new BufferedReader(
		            new FileReader(folderName+fileName+"-test-edited.arff")));
		  System.out.println("전체 데이터 건 수 : " + (train.size() + test.size()) + 
		   " 훈련데이터 건 수 : " + train.size() + 
		  " 테스트 건 수 : " + test.size());
	 }
	 
	 // main 함수
	 public static void main(String[] args) throws Exception {
		  W2_L4_TextMining_Base obj = new W2_L4_TextMining_Base();
		  
		  obj = new W2_L4_TextMining_Base("ReutersGrain");
		  obj.suppliedTestSet(new J48());
		
		  obj = new W2_L4_TextMining_Base("ReutersGrain");
		  obj.suppliedTestSet(new NaiveBayesMultinomial());
	 }

	 // FilteredClassifier 모델적용
	 public void suppliedTestSet(Classifier classifier) throws Exception{
		  // 2) class assigner
		  train.setClassIndex(train.numAttributes()-1);
		  test.setClassIndex(test.numAttributes()-1);
		  
		  // 3) evaluation, classifier setting  
		  Evaluation eval=new Evaluation(train);
		
		  if (this.model == null)
		   this.model = this.setModel(classifier);
		  
		  // 4) model run 
		  model.buildClassifier(train);
		  
		  // 5) evaluate
		  eval.evaluateModel(model, test);
		  
		  // 6) print Result text		  
		  System.out.print(this.getModelName(classifier) + " : " );
		  System.out.print("정분류율 : " + 
		                   String.format("%.2f", eval.pctCorrect()) + " %");
		  System.out.println(", Weighted Area Under ROC : " + 
                  String.format("%.2f",eval.weightedAreaUnderROC()) + "\n\n");
		  System.out.println(eval.toMatrixString());    
		  this.printPctCorrect2ndLabel(eval.confusionMatrix());
	 }
	 
	 // FilteredClassifier 설정
	 public FilteredClassifier setModel(Classifier classifier){
		  FilteredClassifier model = new FilteredClassifier();
		  model.setClassifier(classifier);
		  model.setFilter(new StringToWordVector());
		  
		  return model;
	 }
	 
	 // 2번째 라벨 특이도 출력
	 public void printPctCorrect2ndLabel(double[][] confusionMatrix){
		 
		 double FP = confusionMatrix[1][0];
		 double TN = confusionMatrix[1][1];

		 System.out.println("특이도 : " + 
				            TN + " / " + (FP+TN) + " = " + 
		                    String.format("%.2f", TN/(FP+TN) *100 ) + 
		                    " % ");
	 }
	 
	 
	 // 모델명 획득
	 public String getModelName(Classifier model){
		  String modelName = "";
		  if ( model instanceof  Logistic)
		   modelName = "Logistic";
		  else if ( model instanceof  J48)
		   modelName = "J48";
		  else if ( model instanceof  IBk)
		   modelName = "IBk";
		  else if ( model instanceof  NaiveBayesMultinomial)
		   modelName = "NaiveBayesMultinomial";
		  return modelName;
	 }
	 
	 
	 public W2_L4_TextMining_Base(){

			try{
				Classifier model = new J48();
				model.buildClassifier(null);
			}catch(Exception e){
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
			}
	 }
}