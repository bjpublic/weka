package _2_more;

import java.io.*;
import java.util.Random;

import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.*;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.updater.Adam;

public class W5_L6_DL4J_2_Mnist_Model {

	public static void main(String args[]) throws Exception{
		W5_L6_DL4J_2_Mnist_Model obj = new W5_L6_DL4J_2_Mnist_Model();
		obj.skip();
		obj.runWekadl4j_mnist();
	}

	/*****************************
	 * runWekaDl4j - mnist
	 *****************************/
	public void runWekadl4j_mnist() throws Exception{
		int seed = 1;
		// Set up the MLP classifier
		Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
		clf.setSeed(1);
		clf.setNumEpochs(10);

		// Load the arff file
		Instances data = new Instances(new FileReader("C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j\\datasets\\nominal\\mnist.meta.minimal.arff"));

		long start = System.currentTimeMillis();
		
		if( this.loadModelfromKF() == null ){
			this.makeModel(clf, data);
		}else{
			int trainSize = (int)Math.round(data.numInstances() * 90 / 100);
			int testSize = data.numInstances() - trainSize;
			data.randomize(new java.util.Random(seed));
			
			Instances train = new Instances (data, 0 ,trainSize);
			Instances test  = new Instances (data, trainSize ,testSize);
			clf = (Dl4jMlpClassifier)this.loadModelfromKF();
//
			// 2) class assigner
			data.setClassIndex(data.numAttributes()-1);
			train.setClassIndex(train.numAttributes()-1);
			test. setClassIndex(test. numAttributes()-1);
			
//			// Evaluate the network
			Evaluation eval = new Evaluation(train);
			
			/***************************
			// 이미 평가된 모델이므로 평가단계 불필요
			 ***************************/
//			eval.crossValidateModel(clf, data, numfolds, new Random(seed));

//			// 4) model run 
			/***************************
			// 이미 평가된 모델이므로 생성단계 불필요
			 ***************************/
//			clf.buildClassifier(train);

			// 5) evaluate
			eval.evaluateModel(clf, test);

			System.out.println("\t DL4J 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
					           ", 정분류 건수 : " + (int)eval.correct() + 
					           ", 정분류율 : " + String.format("%.2f",eval.correct() / eval.numInstances() * 100) +" %" 
					           ); 	
			System.out.println(eval.toMatrixString());
		}	
		

		long end = System.currentTimeMillis();
		
		System.out.println("경과시간 : " + (end-start)/1000 + " 초");
	}	
		
	public void makeModel(Dl4jMlpClassifier clf, Instances data ) throws Exception{	
		int seed = 1;
		int numfolds = 10;
		int trainSize = (int)Math.round(data.numInstances() * 90 / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);

		// Load the image iterator
		ImageInstanceIterator imgIter = new ImageInstanceIterator();
		imgIter.setImagesLocation(new File("C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j\\datasets\\nominal\\mnist-minimal"));
		imgIter.setHeight(28);
		imgIter.setWidth(28);
		imgIter.setNumChannels(1);
//		imgIter.setTrainBatchSize(16);
		clf.setInstanceIterator(imgIter);
//		clf.setDataSetIterator(imgIter);

		// Setup the network layers
		// First convolution layer, 8 3x3 filter 
		ConvolutionLayer convLayer1 = new ConvolutionLayer();
		convLayer1.setKernelSizeX(3);
		convLayer1.setKernelSizeY(3);
		convLayer1.setStrideRows(1);
		convLayer1.setStrideColumns(1);
//		convLayer1.setStrideX(1);
//		convLayer1.setStrideY(1);
		convLayer1.setActivationFunction(new ActivationReLU());
		convLayer1.setNOut(8);

		// First maxpooling layer, 2x2 filter
		SubsamplingLayer poolLayer1 = new SubsamplingLayer();
		poolLayer1.setPoolingType(PoolingType.MAX);
		poolLayer1.setKernelSizeX(2);
		poolLayer1.setKernelSizeY(2);
		poolLayer1.setStrideRows(1);
		poolLayer1.setStrideColumns(1);
//		poolLayer1.setStrideX(1);
//		poolLayer1.setStrideY(1);

		// Second convolution layer, 8 3x3 filter
		ConvolutionLayer convLayer2 = new ConvolutionLayer();
		convLayer2.setKernelSizeX(3);
		convLayer2.setKernelSizeY(3);
		convLayer2.setStrideRows(1);
		convLayer2.setStrideColumns(1);
//		convLayer2.setStrideX(1);
//		convLayer2.setStrideY(1);
		convLayer2.setActivationFunction(new ActivationReLU());
		convLayer2.setNOut(8);

		// Second maxpooling layer, 2x2 filter
		SubsamplingLayer poolLayer2 = new SubsamplingLayer();
		poolLayer2.setPoolingType(PoolingType.MAX);
		poolLayer2.setKernelSizeX(2);
		poolLayer2.setKernelSizeY(2);
		poolLayer2.setStrideRows(1);
		poolLayer2.setStrideColumns(1);
//		poolLayer2.setStrideX(1);
//		poolLayer2.setStrideY(1);

		// Output layer with softmax activation
		OutputLayer outputLayer = new OutputLayer();
		outputLayer.setActivationFunction(new ActivationSoftmax());
		outputLayer.setLossFn(new LossMCXENT());

		// Set up the network configuration
		NeuralNetConfiguration nnc = new NeuralNetConfiguration();
		nnc.setUpdater(new Adam());
		clf.setNeuralNetConfiguration(nnc);

		// Set the layers
		clf.setLayers(new Layer[]{convLayer1, poolLayer1, convLayer2, poolLayer2, outputLayer});

		// 2) class assigner
		data.setClassIndex(data.numAttributes()-1);
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) Evaluate the network
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(clf, train, numfolds, new Random(seed));

		// 4) model run 
		clf.buildClassifier(train);
		
		/*****************************
		// 5) save model
		 *****************************/
		this.saveModel(clf);

		// 6) evaluate
		eval.evaluateModel(clf, test);

		System.out.println("model saved");
		
		System.out.println("\t DL4J 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.2f",eval.correct() / eval.numInstances() * 100) +" %" 
				           ); 	
		System.out.println(eval.toMatrixString());
		
	}
	
	public void skip(){

		try{
			Classifier model = new J48();
			model.buildClassifier(null);
		}catch(Exception e){
			System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		}
	}

	public void saveModel(Classifier model) throws Exception{
		String path = "C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j";
		weka.core.SerializationHelper.write(path +"\\model\\mnist_model", model);
	}
	
	public Object loadModel() throws Exception{
		String path = "C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j";
		try{
			return weka.core.SerializationHelper.read(path +"\\model\\mnist_model");
		}catch (Exception e){
			return null;			
		}
	}
	
	public Object loadModelfromKF() throws Exception{
		String path = "C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j";
		try{
			return weka.core.SerializationHelper.read(path +"\\model\\_10_10_Dl4jMlpClassifier.model");
		}catch (Exception e){
			return null;			
		}
	}

}
