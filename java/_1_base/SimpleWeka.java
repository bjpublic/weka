package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

public class SimpleWeka {
	
	public static void main(String args[]) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		// 1) data loader (훈련 데이터와 텍스트 데이터를 기본 8:2 로 분리한다.)
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader("D:\\Weka-3-9\\data\\iris.arff")));
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		RandomForest model=new RandomForest();

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);

		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) random forest run 
		model.buildClassifier(train);
//		model.setOptions(Utils.splitOptions("-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"));
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println(model);                  // model info
		System.out.println(eval.toSummaryString()); // === Evaluation result ===
		System.out.println(eval.toMatrixString());  // === Confusion Matrix === 
	}

}
