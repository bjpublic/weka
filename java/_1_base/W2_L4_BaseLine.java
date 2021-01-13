package _1_base;

import java.io.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W2_L4_BaseLine {
	
	public W2_L4_BaseLine(){
		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}
	
	public static void main(String args[]) throws Exception{
		W2_L4_BaseLine obj = new W2_L4_BaseLine();		
		/**********************************************************
		 * 기준분류기 ZeroR 을 필두로 4개 분류기 정확도 출력 시작
		 **********************************************************/
		System.out.print("ZeroR");obj.baseLine(new ZeroR(),66);
		System.out.print("J48");obj.baseLine(new J48(),66);
		System.out.print("NaiveBayes");obj.baseLine(new NaiveBayes(),66);
		System.out.print("IBk");obj.baseLine(new IBk(),66);
		System.out.print("PART");obj.baseLine(new PART(),66);
		/**********************************************************
		 * 기준분류기 ZeroR 을 필두로 4개 분류기 정확도 출력 시작
		 **********************************************************/
	}
	
	public void baseLine(Classifier model, int percent) throws Exception{
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\supermarket.arff")));

		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(1));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) object creation  
		Evaluation eval=new Evaluation(train);
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + eval.numInstances() + ", 정분류 건수 : " + eval.correct() + ", 분류정확도 : " + eval.correct() / eval.numInstances() * 100 +" %"); 	
	}
}
