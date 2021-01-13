package _1_base;

import java.io.*;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W2_L56_CrossValidation {
	
	public W2_L56_CrossValidation(){
		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}

	public static void main(String args[]) throws Exception{
		W2_L56_CrossValidation obj = new W2_L56_CrossValidation();
		
		/** holdout seed 1씩 증가 호출 **/
		System.out.println("90% holdouts , ");
		double sum[] = new double[10];
		for(int x = 1 ; x <= 10 ; x ++)
			sum [x-1] = obj.holdout(x); // 정분류율 누적
		obj.aggregateValue(sum); // 통계량 출력
		
		System.out.println("");
		
		/** 10 교차검증 seed 1씩 증가 호출 **/
		System.out.println("10 교차검증 , ");
		sum = new double[10];
		for(int x = 1 ; x <= 10 ; x ++)
			sum [x-1] = obj.crossValidation(x); // 정분류율 누적
		obj.aggregateValue(sum); // 통계량 출력
	}

	/**
	 * common-math jar 다운로드 위치 : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public void aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  sumObj.addValue(sum[i]); 

		System.out.println("평균 : " + String.format("%.1f",aggregate.getMean()) + " %, 분산 : " + String.format("%.1f",aggregate.getStandardDeviation()));
	}

	public double crossValidation(int seed) throws Exception{

		int numfolds = 10;
		int numfold = 0;
		// 1) data loader 
		Instances data=new Instances(
				        new BufferedReader(
				        new FileReader("D:\\Weka-3-9\\data\\diabetes.arff")));

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		eval.crossValidateModel(model, train, numfolds, new Random(seed)); 

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
				           ", seed : " + seed ); 	
		// 7) 분류정확도 반환
		return eval.correct() / eval.numInstances() * 100;
	}
	
	public double holdout(int seed) throws Exception{
		// 1) data loader 
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader("D:\\Weka-3-9\\data\\diabetes.arff")));
		int trainSize = (int)Math.round(data.numInstances() * 90 / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		           ", 정분류 건수 : " + (int)eval.correct() + 
		           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
		           ", seed : " + seed ); 	
		
		// 7) 분류정확도 반환
		return eval.correct() / eval.numInstances() * 100;
	}	
	
}
