package _2_more;

import java.io.*;
import java.util.Random;

import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.*;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.lossfunctions.*;
import weka.dl4j.updater.Adam;

public class W5_L2_wekaDL4J_1_iris {

	public static void main(String args[]) throws Exception{
		W5_L2_wekaDL4J_1_iris obj = new W5_L2_wekaDL4J_1_iris();
		obj.skip();

		new W5_L2_wekaDL4J_1_iris().run(new J48(), "iris");
		new W5_L2_wekaDL4J_1_iris().runWekaDl4j("iris");
		new W5_L2_wekaDL4J_1_iris().run(new LinearRegression() , "cpu");
		new W5_L2_wekaDL4J_1_iris().runWekaDl4j("cpu");
	}


	/*****************************
	 * runWekaDl4j - iris 
	 *****************************/
	public void runWekaDl4j(String fileName) throws Exception{
		// Load the iris dataset and set its class index
		// 1) data loader 
		String folderName = "D:\\Weka-3-9\\data\\";
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader(folderName+fileName+".arff")));
		data.setClassIndex(data.numAttributes() - 1);
		
		// Create a new Multi-Layer-Perceptron classifier
		Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
		// Set a seed for reproducable results
		clf.setSeed(1);
		
		// Define the output layer
		OutputLayer outputLayer = new OutputLayer();
//		outputLayer.setActivationFunction(new ActivationSoftmax()); // 회귀분석 불가로 변경
//		outputLayer.setLossFn(new LossMCXENT()); // 회귀분석 불가로 변경

		outputLayer.setActivationFunction(new ActivationSigmoid());
		outputLayer.setLossFn(new LossMSLE());

		NeuralNetConfiguration nnc = new NeuralNetConfiguration();
		nnc.setUpdater(new Adam());

		// Add the layers to the classifier
		clf.setLayers(new Layer[]{outputLayer});
		clf.setNeuralNetConfiguration(nnc);

		// Evaluate the network
		Evaluation eval = new Evaluation(data);
		int numFolds = 10;
		eval.crossValidateModel(clf, data, numFolds, new Random(1));

		System.out.println("**************** weka dl4j output ****************");
		System.out.println(eval.toSummaryString());
		
		System.out.println(clf);
		System.out.println("**************** weka dl4j output ****************");
		
	}	
		

	public void run(Classifier model, String fileName) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		// 1) data loader 
		String folderName = "D:\\Weka-3-9\\data\\";
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader(folderName+fileName+".arff")));
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text 
		System.out.println("**************** run output ****************");
		System.out.println(eval.toSummaryString());
		
		System.out.println(model);		
		System.out.println("**************** run output ****************");
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
