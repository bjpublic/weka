package _2_more;

import java.io.*;
import java.util.Random;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.trees.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.RnnSequenceClassifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.dl4j.GradientNormalization;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.iterators.instance.sequence.text.rnn.RnnTextEmbeddingInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class W5_L2_wekaDL4J_3_IMDB {
	Instances data = null;
	Instances train = null;
	Instances test  = null;

	public static void main(String args[]) throws Exception{
		W5_L2_wekaDL4J_3_IMDB obj = new W5_L2_wekaDL4J_3_IMDB();
		obj.skip();
		obj.setInstances();
		obj.runWekaDL4J_IMDB();
		obj.runNBMultinomial();
	}
		

	/**
	 * 
	 * RnnSequenceClassifier 와 NaiveBayesMultinomial 성능을 비교하기 위해
	 * 동일한 훈련 및 테스트 데이터를 생성한다.
	 * 
	 * **/
	
	public void setInstances() throws Exception{
//		int seed = 1;
//		int numfolds = 10;
//		int numfold = 0;
		String path = "C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j";
		this.data = new Instances(new FileReader(path+"\\datasets\\imdb-sentiment-2011.arff"));
						
		// 1 % (500건) 데이터세트로 조정 (50,000 만건 실행시 메모리 부족 에러)
		data.setClassIndex(1);
		
		train = this.resample(data,0.9)
//				.trainCV(numfolds, numfold, new Random(seed))
				;
		test  = this.resample(data,0.1)
//				.testCV (numfolds, numfold)
				;
		
	}

	/*****************************
	 * runWekaDL4J_IMDB - RnnTextEmbeddingInstanceIterator
	 *****************************/
	public void runWekaDL4J_IMDB() throws Exception{
		long startime = System.currentTimeMillis();
		String path = "C:\\Users\\bulle\\wekafiles\\packages\\wekaDeeplearning4j";
		final File modelSlim = new File(path+"\\datasets\\GoogleNews-vectors-negative300-SLIM.bin");

		// Setup hyperparameters
		final int truncateLength = 80;
		final int batchSize = 64;
		final int seed = 1;
		final int numEpochs = 10;
		final int tbpttLength = 20;
		final double l2 = 1e-5;
		final double gradientThreshold = 1.0;
//		final double learningRate = 0.02;

		// Setup the iterator (입력층)
		RnnTextEmbeddingInstanceIterator tii = new RnnTextEmbeddingInstanceIterator();
		tii.setWordVectorLocation(modelSlim); // embedding 파일경로
		tii.setTruncateLength(truncateLength); // 80
		tii.setTrainBatchSize(batchSize); // 64

		// Define the layers (은닉층)
		LSTM lstm = new LSTM();
		lstm.setNOut(64);
		lstm.setActivationFunction(new ActivationTanH());

		// (출력층)
		RnnOutputLayer rnnOut = new RnnOutputLayer();
		
		// Network config
		NeuralNetConfiguration nnc = new NeuralNetConfiguration();
		nnc.setL2(l2); // 1e-5
//		nnc.setUseRegularization(true);
		nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
		nnc.setGradientNormalizationThreshold(gradientThreshold); // 1.0
//		nnc.setLearningRate(learningRate);

		// Initialize the classifier
		RnnSequenceClassifier clf = new RnnSequenceClassifier();
		clf.setSeed(seed); // 1
		clf.setNumEpochs(numEpochs); // 10
		clf.setInstanceIterator(tii);
		clf.settBPTTbackwardLength(tbpttLength); // 64
		clf.settBPTTforwardLength(tbpttLength);  // 64

		// Config classifier
		clf.setLayers(lstm, rnnOut); // 출력층 설정
		clf.setNeuralNetConfiguration(nnc); 
		
		// 2) class assigner
		train.setClassIndex(1);
		test. setClassIndex(1);
		
		System.out.println("full size : " + data.size()+ ", " +
				           "reduced size : " + (train.size() + test.size())+ ", " +
				           "train size : " + train.size()+ ", " +
				           "test size : " + test.size()
				           );
		
		// 3) Evaluate the network
		Evaluation eval = new Evaluation(train);

		// 4) model run 
		clf.buildClassifier(train);

		// 5) evaluate
		eval.crossValidateModel(clf, train, 10, new Random(seed));
		eval.evaluateModel(clf, test);
		
		System.out.print("\t deeplearning4j 실행시간 : " + 
		                   (System.currentTimeMillis() - startime)/1000 + " 초, " +
		                   " DL4J 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.2f",eval.pctCorrect()) +" %" 
				           ); 	
		System.out.println(", Weighted Area Under ROC : " + 
                           String.format("%.2f",eval.weightedAreaUnderROC()));
		System.out.println(eval.toMatrixString());
		System.out.println(clf);
		
	}	
	
	/**
	 * 50,000 건 데이터세트 로드시 메모리 부족으로 에러발생
	 * 1% (500건) 데이터세트로 조정
	 * **/
	public Instances resample(Instances data, double newSampleSizePercent) throws Exception{
		// newSampleSizePercent % 데이터세트로 조정
		Resample supervisedFilter = new Resample();
		
		supervisedFilter.setSampleSizePercent(newSampleSizePercent);
		supervisedFilter.setInputFormat(data);
		Instances reducedData = Filter.useFilter(data, supervisedFilter);	
		
		return reducedData;
	}
			
		 // FilteredClassifier 모델적용

	/**********************************
	 * FilteredClassifier 내에 
	 * StringToWordVector 필터,  
	 * NaiveBayesMultinomial 알고리즘 적용
	 **********************************/
	 public void runNBMultinomial() throws Exception{
		long startime = System.currentTimeMillis();
		int seed = 1;
		
		// 2) class assigner
		train.setClassIndex(1);
		test.setClassIndex(1);
		
		System.out.println("full size : " + data.size()+ ", " +
				   "reduced size : " + (train.size() + test.size())+ ", " +
		           "train size : " + train.size()+ ", " +
		           "test size : " + test.size()
		           );
		  
		// 3) evaluation, classifier setting  
		Evaluation eval=new Evaluation(train);
		
		Classifier model = this.setModel(new NaiveBayesMultinomial());
		  
		// 4) model run 
		model.buildClassifier(train);
		  
		// 5) evaluate
		eval.crossValidateModel(model, train, 10, new Random(seed));
		eval.evaluateModel(model, test);
		 
		// 6) print Result text		  
		System.out.print("NBMultinomial in FilteredClassifier " );
		System.out.print("\t 실행시간 : " + 
                         (System.currentTimeMillis() - startime)/1000 + " 초, " +
				         "정분류율 : " + String.format("%.2f", eval.pctCorrect()) + " %");
		System.out.println(", Weighted Area Under ROC : " + 
                         String.format("%.2f",eval.weightedAreaUnderROC()));
		System.out.println(eval.toMatrixString());    
		this.printPctCorrect2ndLabel(eval.confusionMatrix());
	 }
	 
	 // FilteredClassifier 설정
	 public FilteredClassifier setModel(NaiveBayesMultinomial classifier){
		  // 1) 분류알고리즘 설정
		  FilteredClassifier model = new FilteredClassifier();
		  model.setClassifier(classifier);
		  // 2) 필터설정
		  StringToWordVector word2vector = new StringToWordVector();
		  System.out.println("StringToWordVector 설정값 변경");
		  word2vector.setLowerCaseTokens(true);
		  word2vector.setOutputWordCounts(true);
		  word2vector.setStopwordsHandler(new Rainbow());
		  word2vector.setStemmer(new SnowballStemmer());
		  word2vector.setWordsToKeep(800);
		  model.setFilter(word2vector);
		  
		  return model;
	 }
	 
	 // 2번째 라벨 특이도 출력
	 public void printPctCorrect2ndLabel(double[][] confusionMatrix){
		 
		 double FP = confusionMatrix[1][0];
		 double TN = confusionMatrix[1][1];

		 System.out.println("특이도 : " + 
				            TN + " / " + (FP+TN) + " = " + 
		                    String.format("%.2f", TN/(FP+TN) *100 ) + 
		                    " % ");
	 }	 

	 // 모델명 획득
	 public String getModelName(Classifier model){
		  String modelName = "";
		  if ( model instanceof  Logistic)
		   modelName = "Logistic";
		  else if ( model instanceof  J48)
		   modelName = "J48";
		  else if ( model instanceof  IBk)
		   modelName = "IBk";
		  else if ( model instanceof  NaiveBayesMultinomial)
		   modelName = "NaiveBayesMultinomial";
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
