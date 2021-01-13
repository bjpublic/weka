package _2_more;

import java.io.*;
import java.util.Random;



import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W5_L4_PerformanceOptimization {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W5_L4_PerformanceOptimization (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
	    new W5_L4_PerformanceOptimization("breast-cancer")
	    .optimizer(new IBk(), "K 1 10 10");
	}
	
	/*****************************
	 * crossvalidation
	 *****************************/
	public void optimizer(Classifier model, String optimizerOption) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		int seed = 1;
		  
		// 1) data split		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);

		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		CVParameterSelection optimizer = new CVParameterSelection();		  

		// 4) model run 
		optimizer.setClassifier(model);
		this.setCVParameters(optimizer, optimizerOption);
		eval.crossValidateModel(optimizer, train, numfolds, new Random(seed));
	
		optimizer.buildClassifier(train);

		// 5) evaluate
		eval.evaluateModel(optimizer, test);
		
		// 6) print Result text (분류기 정분류율 출력)
		this.printClassfiedInfo(optimizer, eval);
		
		System.out.println(optimizer);
		System.out.println("제안하는 이웃 군집수 : " + ((IBk)optimizer.getClassifier()).getKNN());
		
		// 7) save model
		this.saveModel(model);
	}

	/****
	 * 
	 * CVParameterSelection 의  CVParameters 설정
	 * 
	 ****/
	public void setCVParameters(CVParameterSelection optimizer, String optimizerOption) throws Exception{
		String[] options = new String[1];
		options[0] = optimizerOption;
		optimizer.setCVParameters(options);
	}
	
	private double printClassfiedInfo(Classifier model, Evaluation eval) throws Exception {
		// 6) print Result text
		System.out.print("\n** " + this.getModelName(model) +  
                         ", 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		                 ", 정분류 건수 : " + (int)eval.correct() + 
		                 ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
		                 ""); 	
		System.out.print(", Weighted Area Under ROC : " + 
        String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
		System.out.println(eval.toMatrixString());
		return 0;
	}

	/**************************************
	 * 7) Print OverFitting 
	**************************************/
	public void printOverFitting(Classifier model, double crossValidation, double useTrainingSet) throws Exception{		
		System.out.println(" ==> Overfitting (difference) :" + 
	                       String.format("%.2f",useTrainingSet - crossValidation) + " % (" + 
				           getModelName(model)+ ")");
		System.out.println("");
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
			modelName = "MultilayerPerceptron";		
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
	
	public void saveModel(Classifier model) throws Exception{
		String path = "D:\\Weka-3-9\\model\\";
		weka.core.SerializationHelper.write(path +"opt_model", model);
	}
	
	public Object loadModel() throws Exception{
		String path = "D:\\Weka-3-9\\model\\";
		return weka.core.SerializationHelper.read(path +"opt_model");
	}
	

}
