package _2_more;

import java.io.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.*;

public class W5_L1_BasePerceptron {

	public static void main(String args[]) throws Exception{
		W5_L1_BasePerceptron obj = new W5_L1_BasePerceptron();
		obj.skip();
		System.out.println("=============================================================================");
		System.out.println("\t 1) BasePerceptron �� �з��� ���� ���з� ��");
		System.out.println("=============================================================================");
		
		obj.getMultiPerceptron("5,10,20");
		
	    Classifier[] models = {new VotedPerceptron(), new SMO(), new J48(), new Logistic(), 
	    		               obj.getMultiPerceptron("a"),obj.getMultiPerceptron("5,10,20")}; 
	    
	    String[] fileNames = {"supermarket"};
	    for (String fileName : fileNames) {
	    	System.out.println("\nfileName : " + fileName);
		    for(Classifier model : models){
				obj.split(model,fileName, 66);
			}// end-for-models	
		}

		System.out.println("=============================================================================");
		System.out.println("\t 1) BasePerceptron");
		System.out.println("=============================================================================");
	}


	/*****************************
	 * split
	 *****************************/
	public void split(Classifier model, String fileName, int percent) throws Exception{
    	double startTime = System.currentTimeMillis();
		int seed = 1;
		// 1) data loader 
		String folderName = "D:\\Weka-3-9\\data\\";
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader(folderName+fileName+".arff")));
		
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
		// �Ʒ�/�׽�Ʈ ������ �и��Ǿ� �����Ƿ� �������� ���ʿ�
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) ���з���, ROC
		this.printClassfiedInfo(model, eval, System.currentTimeMillis() - startTime);
		
	}	
	
	// �ΰ��Ű�� ������ ����
	public MultilayerPerceptron getMultiPerceptron(String hiddenlayers){
		
		MultilayerPerceptron perceptron = new MultilayerPerceptron();
		perceptron.setGUI(false); // true �� �����ϸ� �Ű�� GUI �� �����ȴ�. 
		perceptron.setHiddenLayers(hiddenlayers);
		
		return perceptron;
	}
	
	private double printClassfiedInfo(Classifier model, Evaluation eval, double runTime) throws Exception {
		// 6) print Result text
		System.out.print(" �з���� ������ �� �� : " + (int)eval.numInstances() + 
		                 ", ���з� �Ǽ� : " + (int)eval.correct() + 
		                 ", ���з��� : " + String.format("%.1f",eval.pctCorrect()) +" %" +
		                 ""); 	
		System.out.print(", ROC : " + 
                         String.format("%.2f",eval.weightedAreaUnderROC()) + 
                         ", ����ð� :" + String.format("%.2f",runTime/1000) +" sec " +
                         " ** " + this.getModelName(model) + "\n");
//		System.out.println(eval.toMatrixString());
		return 0;
	}
	
	public String getModelName(Classifier Classifier) throws Exception{
		String modelName = "";
		if(Classifier instanceof VotedPerceptron){
			modelName = "VotedPerceptron";
		}else if(Classifier instanceof SMO){
			modelName = "SMO";		
		}else if(Classifier instanceof J48){
			modelName = "J48";		
		}else if(Classifier instanceof Logistic){
			modelName = "Logistic";		
		}else if(Classifier instanceof MultilayerPerceptron){
			modelName = "MultilayerPerceptron, hidden layers nodes :" + 
		         ((MultilayerPerceptron)Classifier).getHiddenLayers();		
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
