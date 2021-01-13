package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;

public class W3_L3_NaiveBayes {

	public static void main(String args[]) throws Exception{
		W3_L3_NaiveBayes obj = new W3_L3_NaiveBayes();
		
		System.out.println("weather.norminal : ");
		obj.weatherNominalNavieBayes(new NaiveBayes());  
	}

	public void weatherNominalNavieBayes(Classifier obj) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\weather.nominal.arff")));
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=obj; // 매개변수로 받은 분류기 대입		
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" 
				           + ", 분류기 : " + obj + "교차검증 모델평가" 
				           ); 	
//		System.out.println(model);
	}
	
}
