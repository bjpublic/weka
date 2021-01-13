package _2_more;

import java.io.*;
import java.util.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;

public class W4_L3_Scheme_independent_with_FilteredClassifier {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L3_Scheme_independent_with_FilteredClassifier (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		for(int x=0 ; x < 4 ; x++)
			new W4_L3_Scheme_independent_with_FilteredClassifier("ReutersGrain-train-edited")
				.scheme_independent(x, new NaiveBayes());

		for(int x=0 ; x < 4 ; x++)
			new W4_L3_Scheme_independent_with_FilteredClassifier("ReutersGrain-train-edited")
				.scheme_independent(x, new NaiveBayesMultinomial());

		new W4_L3_Scheme_independent_with_FilteredClassifier("ReutersGrain-train-edited")
			.metaFilteredClassifier(5, new NaiveBayesMultinomial());

		new W4_L3_Scheme_independent_with_FilteredClassifier("ReutersGrain-train-edited")
			.metaFilteredClassifier(6, new NaiveBayesMultinomial());
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
				System.out.print("\n ====== 1) 일반적인 경우 실행 :");
				attrSelectInstances = this.setWordToVector(false);
				break;
			case 1:
				System.out.print("\n ====== 2) AttributeSelect panel :");
				attrSelectInstances = this.setWordToVector(false);
				attrSelectInstances = this.panelOfselectAttribute(attrSelectInstances);
				classIndex = attrSelectInstances.numAttributes()-1;
				break;
			case 2:
				System.out.print("\n ====== 3) meta 알고리즘 :");
				attrSelectInstances = this.setWordToVector(false);
				model = this.metaClassifier(model);
				break;
			case 3:
				System.out.print("\n ====== 4) meta 알고리즘 (true) :");
				attrSelectInstances = this.setWordToVector(true);
				model = this.metaClassifier(model);
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
//		System.out.println(eval.toMatrixString());
		this.printPctCorrect2ndLabel(eval.confusionMatrix());
		
	}

	public void metaFilteredClassifier(int testCase, Classifier model) throws Exception{		
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		int classIndex = this.data.numAttributes()-1;
				
		// 1) data split 
		Instances train = this.data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = this.data.testCV (numfolds, numfold);

		// 2) class assigner
		this.data.setClassIndex(classIndex);
		train.setClassIndex(classIndex);
		test. setClassIndex(classIndex);
		
		// 3) eval setting  
		Evaluation eval=new Evaluation(train);
		Classifier meta = null;

		switch(testCase){
			case 5:
				System.out.print("\n ====== 5) FilteredClassifier + filter : MultiFilter :");
				meta = this.setFilteredClassifier_MultiFilter(model);
				break;
			case 6:
				System.out.print("\n ====== 6) FilteredClassifier + classifier : AttributeSelectedClassifier :");
				meta = this.setFilteredClassifier_AttributeSelectedClassifier(model);
				break;
		}		
		eval.crossValidateModel(meta, train, numfolds, new Random(seed));

		// 4) model run 
		meta.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(meta, test);
		
		// 6) print Result text
		System.out.print("\t" + this.getModelName(meta) +  
		                   ", 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
				           ""); 	
		System.out.print("\t , Weighted Area Under ROC : " + 
                           String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
//		System.out.println(eval.toMatrixString());
		this.printPctCorrect2ndLabel(eval.confusionMatrix());
		
	}

	/*********************************************
	 * FilteredClassifier (filter : MultiFilter) 
	 *********************************************/
	public Classifier setFilteredClassifier_MultiFilter(Classifier newClassifier)
			throws Exception{
		FilteredClassifier meta = new FilteredClassifier();
		
		// Classifier 
		meta.setClassifier(newClassifier);

		Filter[] filters = new Filter[2];

		// 1st StringToWordVector filter
		StringToWordVector word2vector = new StringToWordVector();
		word2vector.setInputFormat(this.data);				
		filters[0] = word2vector;

		// 2nd StringToWordVector filter
		weka.filters.supervised.attribute.AttributeSelection attrSel = null;
		attrSel = new weka.filters.supervised.attribute.AttributeSelection();
		filters[1] = attrSel;		
		
		MultiFilter filter = new MultiFilter();
		filter.setFilters(filters);
		meta.setFilter(filter);
		return meta;
	}
	
	/****************************************************************
	 * FilteredClassifier (classifier : AttributeSelectedClassifier) 
	 ***************************************************************/
	public Classifier setFilteredClassifier_AttributeSelectedClassifier(Classifier newClassifier) throws Exception{
		FilteredClassifier meta = new FilteredClassifier();
		
		// Classifier 
		AttributeSelectedClassifier classifer = null;
		classifer = new AttributeSelectedClassifier();
		classifer.setClassifier(newClassifier);
		
		meta.setClassifier(new AttributeSelectedClassifier());

		// filter : StringToWordVector 
		StringToWordVector word2vector = new StringToWordVector();
		word2vector.setInputFormat(this.data);		
		meta.setFilter(word2vector);
		
		return meta;
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
    
	/****************************
	 * selectAttribute 패널 동작
	 ****************************/
	public Instances panelOfselectAttribute(Instances instance) throws Exception{
		
		// "select attribute" 패널 역할 객체
		AttributeSelection attrSelector = new AttributeSelection();
		
		instance.setClassIndex(0);
				
		// Evaluator, SearchMethod setting
		attrSelector.setEvaluator(new CfsSubsetEval());
		attrSelector.setSearch(new BestFirst());
		
		// RUN Selection of Attribute
		attrSelector.SelectAttributes(instance);
		
		// get selected index count
		this.selectedIdx = attrSelector.selectedAttributes();
		
		// print selected attribute
		this.printSelectedAttribute(attrSelector,instance);
		
		// return selected data (reduced attribute)
		return attrSelector.reduceDimensionality(instance);
	}
	
	public void printSelectedAttribute(AttributeSelection attrSelector,Instances instance) throws Exception{
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
	}

	/****************************
	 * selectAttribute 메타 알고리즘 
	 ****************************/
	public Classifier metaClassifier(Classifier classifier){
				
		// 분류 알고리즘 생성
		AttributeSelectedClassifier attrSelClassifier = null;
		
		// classifier 설정
		attrSelClassifier = new AttributeSelectedClassifier();
		attrSelClassifier.setClassifier(classifier);
		
		// Evaluator, SearchMethod setting
		attrSelClassifier.setEvaluator(new CfsSubsetEval());
		attrSelClassifier.setSearch(new BestFirst());
				
		return attrSelClassifier;
	}

	/****************************
	 * StringToWordVector 전처리 필터 
	 ****************************/
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
