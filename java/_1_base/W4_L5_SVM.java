package _1_base;

import java.io.*;
import java.util.*;


import weka.classifiers.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L5_SVM {
	 
	 public static void main(String args[]) throws Exception{

		W4_L5_SVM obj = new W4_L5_SVM();
		System.out.println("=============================================================================");
		System.out.println("\t 1) 1개 데이터세트와 4개 분류기를 배열로 저장한 후 루핑 호출");
		System.out.println("=============================================================================");
	    String fileNames[] = {"credit-g"}; // 퀴즈에서 사용하는 1개 arff 파일을 배열로 저장
	    Classifier[] models = {new IBk(),new J48(), new Logistic(), new SMO()}; // 퀴즈 문항별 비교할 4개 분류기를 배열로 저장
//		String fileNames[] = {"diabetes"}; Classifier[] models = {new Logistic()};

	    for(String fileName : fileNames){
			for(Classifier model : models){
				System.out.println(fileName + " : ");
				double crossValidation = obj.crossValidataion(fileName,model);
				double useTrainingSet  = obj.useTrainingSet(fileName, model);
				
				obj.printOverFitting(model, crossValidation, useTrainingSet);
			}// end-for-models	
		}// end-for-fileNames	 
	}

	/*****************************
	 * 교차검증
	 *****************************/
	public double crossValidataion(String fileName, Classifier model) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		  
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		data.setClassIndex(data.numAttributes()-1); 
		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		  
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		eval.crossValidateModel(model, train, numfolds, new Random(seed));	  

		// 4) model run 
		model.buildClassifier(train);
		   
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text (분류기 정분류율 및 평균제곱편차 출력)
		return this.printClassfiedInfo(model, eval);
			   			   
	}

	/*****************************
	 * GUI use TrainingSet
	 *****************************/
	public double useTrainingSet(String fileName, Classifier model) throws Exception{
		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		
		// 2) class assigner
		data.setClassIndex(data.numAttributes()-1);
		
		// 3) 검증객체 생성
		Evaluation eval = new Evaluation(data);

		// 4) model run 
		model.buildClassifier(data);
		   
		// 5) evaluate
		eval.evaluateModel(model, data);
		
		// 6) print Result text (분류기 정분류율 및 평균제곱편차 출력)
		return this.printClassfiedInfo(model, eval);
	}
	
	/*****************************
	 * 6) 분류기 정분류율 및 평균제곱편차 출력
	 *****************************/
	public double printClassfiedInfo(Classifier model, Evaluation eval){
		System.out.print("Correctly Classified Instances : " + String.format("%.2f",eval.pctCorrect()));
		System.out.print(", Root mean squared error  :" + String.format("%.2f",eval.rootMeanSquaredError()));
		System.out.println(", (" + getModelName(model) + ")");
		
		return eval.pctCorrect();
	}

	/**************************************
	 * 7) Print OverFitting 
	**************************************/
	public void printOverFitting(Classifier model, double crossValidation, double useTrainingSet) throws Exception{		
		System.out.println(" ==> Overfitting (difference) :" + 
	                       String.format("%.2f",useTrainingSet - crossValidation) + " % (" + 
				           getModelName(model)+ ")");
		System.out.println("");
	}

	/*****************************
	 * Model Name
	 *****************************/
	public String getModelName(Classifier model){
		String modelName = "";
		if ( model instanceof  Logistic)
			modelName = "Logistic";
		else if ( model instanceof  J48)
			modelName = "J48";
		else if ( model instanceof  IBk)
			modelName = "IBk";
		else if ( model instanceof  SMO)
			modelName = "SMO";
		return modelName;
	}
}
