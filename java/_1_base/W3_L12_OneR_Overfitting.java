package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.rules.*;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class W3_L12_OneR_Overfitting {
	
	public W3_L12_OneR_Overfitting(){
		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}
	
	public static void main(String args[]) throws Exception{
		W3_L12_OneR_Overfitting obj = new W3_L12_OneR_Overfitting();
		
		/** weather.numeric 호출 **/
		System.out.println("weather.numeric : ");
		obj.weatherNumericHoldOutOneR(false,6);  // 필터 미적용,  minBucketSize = 6
		obj.weatherNumericHoldOutOneR(true,6);   // 필터 적용,   minBucketSize = 6
		obj.weatherNumericHoldOutOneR(true,1);   // 필터 적용,   minBucketSize = 1
		
		System.out.println("");
		
		/** diabete 호출 **/
		System.out.println("diabete : ");
		obj.diabeteCrossValidationOneR(new ZeroR(),false,6); // zeroR,  crossValidate, minBucketSize = 6
		obj.diabeteCrossValidationOneR(new OneR() ,false,6); // OneR,    crossValidate, minBucketSize = 6
		obj.diabeteCrossValidationOneR(new OneR() ,false,1); // OneR,    crossValidate, minBucketSize = 1
		obj.diabeteCrossValidationOneR(new OneR() ,true,1);  // OneR, Use training set, minBucketSize = 1
	}

	public void weatherNumericHoldOutOneR(boolean isRemove, int minBucketSize) throws Exception{
		int seed = 1;
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\weather.numeric.arff")));
		if(isRemove){
			Remove filter = new Remove();
			filter.setAttributeIndices("1");
			filter.setInputFormat(data);
			data = Filter.useFilter(data, filter);		
		}	
		int trainSize = (int)Math.round(data.numInstances() * 66 / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		OneR model=new OneR();
		/************************
		 * MinBucketSize 설정 
		 ************************/
		model.setMinBucketSize(minBucketSize);

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
				           ", minBucketSize : " + minBucketSize + 
				           ", 분류기 : OneR" + ", 홀드아웃 모델평가"); 	
		System.out.println(model);
	}
	
	public void diabeteCrossValidationOneR(Classifier obj, boolean isUseTrainingSet, int minBucketSize) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\diabetes.arff")));
		Instances train = null;
		Instances test  = null;
		if(isUseTrainingSet){ 
			// 분석대상 데이터를 그대로 훈련/테스트 데이터로 설정 (Use training set)
			train = new Instances(data);
			test  = new Instances(data);			
		}else{ 
			// crossValidation
			train = data.trainCV(numfolds, numfold, new Random(seed));
			test  = data.testCV (numfolds, numfold);
		}
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model = obj;
		if(obj instanceof OneR){
			/************************
			 * MinBucketSize 설정 시작
			 ************************/
			((OneR)model).setMinBucketSize(minBucketSize);
		}		
		if(!isUseTrainingSet) // Use Training set Only (훈련데이터 限) 아닌 경우만 실행
			eval.crossValidateModel(model, train, numfolds, new Random(seed)); 

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		           ", 정분류 건수 : " + (int)eval.correct() + 
		           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
		           ", minBucketSize : " + minBucketSize + 
		           ", 분류기 : " + ((obj instanceof ZeroR)?"ZeroR":"OneR") +
		           ", " + ((isUseTrainingSet)?"Use Training set Only (훈련데이터 限)":"crossvalidation (교차검증)") + " 모델평가"  
		           );
		System.out.println(model);
	}	
	
}
