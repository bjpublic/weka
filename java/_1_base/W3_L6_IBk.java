package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddNoise;

public class W3_L6_IBk {

	public static void main(String args[]) throws Exception{
		W3_L6_IBk obj = new W3_L6_IBk();
		String fileName= "glass";
		System.out.println(fileName + " : ");
		
		// 매개변수 설명 : 파일명, kNN 최대값, crossValidate = true 설정, Noise Percentage 설정
		obj.glassIBk(fileName,100,true,0);  
		obj.glassIBk(fileName,100,true,10);  
		obj.glassIBk(fileName,100,true,20);  
		obj.glassIBk(fileName,100,true,30);  
		obj.glassIBk(fileName,100,true,40);  
		obj.glassIBk(fileName,100,true,50);  

		for(int i=1 ; i <10 ; i++) obj.glassIBk(fileName,10*i,true,50);  
	}

	public void glassIBk(String fileName, int k, boolean isCrossValidate, int percentage) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		/*****************************
		 * addNoise 필터 적용 시작
		 *****************************/
		AddNoise filter = new AddNoise();
		filter.setPercent(percentage);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		/*****************************
		 * addNoise 필터 적용 종료
		 *****************************/
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		IBk model=new IBk(); 		
		/**********************************
		 * crossValidate, k, seed 값 변경 시작 
		 **********************************/
		model.setCrossValidate(isCrossValidate);
		model.setKNN(k);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));
		/**********************************
		 * crossValidate, k, seed 값 변경 종료 
		 **********************************/
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" 
				           + ", 분류기 : IBK , k =" + k +" , isCrossValidate = " +  isCrossValidate
				           + " , percentage = " + percentage + " , 제안하는 군집수 : " + model.getKNN()); 	
//		System.out.println(model);
	}
	
}
