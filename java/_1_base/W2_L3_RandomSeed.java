package _1_base;

import java.io.*;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.*;
import weka.core.*;

public class W2_L3_RandomSeed {
	
	double correctRatio = 0.0;
	public static void main(String args[]) throws Exception{
		W2_L3_RandomSeed obj = new W2_L3_RandomSeed();
		double sum[] = new double[10];
		/**********************************************************
		 * RandomSeed를 1씩 증가시켜 정확도를 출력후 성능평균 산출 시작
		 **********************************************************/
		for(int x=1 ; x<=10 ; x++){
			System.out.print("90% split, RandomSeed = " + x);
			sum[x-1] = obj.randomSeed(90,x);
		}	
		obj.aggregateValue(sum);
		/**********************************************************
		 * RandomSeed를 1씩 증가시켜 정확도를 출력후 성능평균 산출 종료
		 **********************************************************/
	}
	
	public double randomSeed(int percent, int seed) throws Exception{
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\segment-challenge.arff")));

		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) learn and evaluate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		this.correctRatio += eval.correct() / eval.numInstances() * 100; // 분류정확도 누적
		System.out.println("\t분류대상 테스트 데이터 건 수 : " + eval.numInstances() + ", 정분류 건수 : " + eval.correct() + ", 분류정확도 : " + eval.pctCorrect() +" %"); 	
	
		return eval.pctCorrect(); // 정분류율 반환
	}
	
	/**
	 * common-math jar 다운로드 위치 : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public void aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  sumObj.addValue(sum[i]); 

		System.out.println("평균 : " + String.format("%.1f",aggregate.getMean()) + " %, 분산 : " + String.format("%.1f",aggregate.getStandardDeviation())  + " %");
	}
}
