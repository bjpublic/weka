package _2_more;

import java.io.*;
import java.util.*;
import weka.attributeSelection.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class W4_L3_Scheme_independent {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L3_Scheme_independent (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		for(int x=0 ; x < 4 ; x++)
			new W4_L3_Scheme_independent("ReutersGrain-train-edited")
				.scheme_independent(x, new NaiveBayes());
		

		for(int x=0 ; x < 4 ; x++)
			new W4_L3_Scheme_independent("ReutersGrain-train-edited")
				.scheme_independent(x, new NaiveBayesMultinomial());
	}
	
	public void scheme_independent(int testCase, Classifier model) throws Exception{		
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		int classIndex = 0;
		
		Instances attrSelectInstances = null;
		
		switch(testCase){
			case 0:
				this.skip();
				System.out.println("\n ====== 1) AttributeSelect component :");
				attrSelectInstances = this.setWordToVector(false);
				attrSelectInstances = this.attrSelComponent(model,attrSelectInstances);
				classIndex = attrSelectInstances.numAttributes()-1;
				break;
			case 1:
				System.out.print("\n ====== 2) 일반적인 경우 실행 :");
				attrSelectInstances = this.setWordToVector(false);
				break;
			case 2:
				System.out.print("\n ====== 3) AttributeSelectedClassifier :");
				attrSelectInstances = this.setWordToVector(false);
				model = this.attrSelClassifier(model);
				break;
			case 3:
				System.out.print("\n ====== 4) AttributeSelectedClassifier (stringToVector lowCase = true)");
				attrSelectInstances = this.setWordToVector(true);
				model = this.attrSelClassifier(model);
				break;
		}
		
		// 1) data split 
		Instances train = attrSelectInstances.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = attrSelectInstances.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(classIndex);
		test. setClassIndex(classIndex);
		
		// 3) eval setting  
		Evaluation eval=new Evaluation(train);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.print("\t" + this.getModelName(model) +  
		                   ", 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
				           ""); 	
		System.out.print("\t , Weighted Area Under ROC : " + 
                           String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
		System.out.println(eval.toMatrixString());
		this.printPctCorrect2ndLabel(eval.confusionMatrix());
		
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
    
	/**
	 * attrSelComponent
	 * */
	public Instances attrSelComponent(Classifier classifier, Instances instance) throws Exception{
		
		// "select attribute" 패널 역할 객체
		AttributeSelection attrSelector = new AttributeSelection();
		
		instance.setClassIndex(0);
		
		// BestFirst default
		
		// WrapperSubsetEval
		CfsSubsetEval subset = new CfsSubsetEval();

		// BestFirst, WrapperSubsetEval setting
		attrSelector.setEvaluator(subset);
		
		// RUN Selection of Attribute
		attrSelector.SelectAttributes(instance);
		
		// get selected index count
		this.selectedIdx = attrSelector.selectedAttributes();
		
		// print index of selected attributes
		for (int x=0 ; x < this.selectedIdx.length ; x++) {			
			// get selected attribute name
			Attribute attr = attrSelector
					        .reduceDimensionality(instance)
					        .attribute(x);
			String attrName= attr.name();
			System.out.println("\t\t selected attribute index : " 
			                   + (this.selectedIdx[x]+1) 
			                   + " : " + attrName);
		}
		
		// return selected data (reduced attribute)
		return attrSelector.reduceDimensionality(instance);
	}
	
	public Classifier attrSelClassifier(Classifier classifier){
				
		// 분류 알고리즘 생성
		AttributeSelectedClassifier attrSelClassifier = null;
		
		// 상위 classifier 설정
		attrSelClassifier = new AttributeSelectedClassifier();
		attrSelClassifier.setClassifier(classifier);
		
		// evaluator 의 classifier 설정
		CfsSubsetEval subset = new CfsSubsetEval();
		attrSelClassifier.setEvaluator(subset);
				
		return attrSelClassifier;
	}
	

	 public Instances setWordToVector(boolean lowcase) throws Exception{
		 
		  // 필터설정
		  StringToWordVector word2vector = new StringToWordVector();
		  word2vector.setLowerCaseTokens(lowcase);
		  word2vector.setInputFormat(this.data);		  
		  
		  return Filter.useFilter(this.data, word2vector);
	 }
	

	public String getModelName(Classifier Classifier) throws Exception{
		String modelName = "";
		if(Classifier instanceof NaiveBayesMultinomial){
			modelName = "NaiveBayesMultinomial";
		}else if(Classifier instanceof NaiveBayes){
			modelName = "NaiveBayes";		
		}else if(Classifier instanceof AttributeSelectedClassifier){
			modelName = "AttributeSelectedClassifier";		
		}
		
		return modelName;
	}
	 
	 public void skip(){

			try{
				Classifier model = new J48();
				model.buildClassifier(null);
			}catch(Exception e){
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
			}
	 }

}
