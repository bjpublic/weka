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

public class W5_L2_wekaDL4J_2_Mnist {

	public static void main(String args[]) throws Exception{
		W5_L2_wekaDL4J_2_Mnist obj = new W5_L2_wekaDL4J_2_Mnist();
		obj.skip();
		obj.runWekadl4j_mnist();
	}

	/*****************************
	 * runWekaDl4j - mnist
	 *****************************/
	public void runWekadl4j_mnist() throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		// Set up the MLP classifier
		Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
		clf.setSeed(1);
		clf.setNumEpochs(10);

		// Load the arff file
		Instances data = new Instances(new FileReader("C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j\\datasets\\nominal\\mnist.meta.minimal.arff"));

		Instances train = data.trainCV(numfolds, numfold, new Random(42));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		data.setClassIndex(data.numAttributes()-1);
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);


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

		// Evaluate the network
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(clf, data, numfolds, new Random(seed));

		// 4) model run 
		clf.buildClassifier(data);

		// 5) evaluate
		eval.evaluateModel(clf, test);

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
}
