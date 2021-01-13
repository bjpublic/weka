package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.*;
import weka.core.*;

public class W2_L2_TrainTest {

	public static void main(String args[]) throws Exception{
		W2_L2_TrainTest obj = new W2_L2_TrainTest();
		System.out.print("suppliedTestSet , ");
		obj.suppliedTestSet();
		System.out.print("90% split, ");
		obj.split(90);
		System.out.print("66% split, ");
		obj.split(66);
	}

	public void suppliedTestSet() throws Exception{

		// 1) data loader 
		/**********************************************************
		 * 1-1) �Ʒ�/�׽�Ʈ �����͸� ������ �ҷ��� ����
		 **********************************************************/
		Instances train=new Instances(
				        new BufferedReader(
				        new FileReader("D:\\Weka-3-9\\data\\segment-challenge.arff")));
		Instances test =new Instances(
			            new BufferedReader(
			            new FileReader("D:\\Weka-3-9\\data\\segment-test.arff")));
		/**********************************************************
		 * 1-1) �Ʒ�/�׽�Ʈ �����͸� ������ �ҷ��� ���� ����
		 **********************************************************/
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) holdout setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		//eval.crossValidateModel(model, train, numfolds, new Random(seed)); --> �Ʒ�/�׽�Ʈ ������ �и��Ǿ� �����Ƿ� ��������  ���ʿ�

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("��ü ������ �Ǽ� : " + (train.size()+test.size()) + 
				           ", �Ʒ� ������ �� �� : " + train.size() + 
				           ", �׽�Ʈ ������ �� �� : " + test.size());
		System.out.println(eval.toSummaryString()); // === Evaluation result ===
	}
	
	public void split(int percent) throws Exception{
		int seed = 1;
		// 1) data loader 
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader("D:\\Weka-3-9\\data\\segment-challenge.arff")));
		/**********************************************************
		 * 1-1) ���� �����͸� �ҷ��� �� �Ʒ�/�׽�Ʈ �����ͷ� �и� ����
		 **********************************************************/
		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		/**********************************************************
		 * 1-1) ���� �����͸� �ҷ��� �� �Ʒ�/�׽�Ʈ �����ͷ� �и� ����
		 **********************************************************/
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) holdout setting  
		Evaluation eval=new Evaluation(train);
		Classifier model=new J48();
		//eval.crossValidateModel(model, train, numfolds, new Random(seed)); --> �Ʒ�/�׽�Ʈ ������ �и��Ǿ� �����Ƿ� �������� ���ʿ�
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("��ü ������ �Ǽ� : " + (data.size()) + 
		           ", �Ʒ� ������ �� �� : " + train.size() + 
		           ", �׽�Ʈ ������ �� �� : " + test.size());
		System.out.println(eval.toSummaryString()); // === Evaluation result ===
	}	
}
