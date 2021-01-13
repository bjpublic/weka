package _2_more;

import java.io.*;
import weka.classifiers.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class W5_L5_MoreArff {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W5_L5_MoreArff (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
	    new W5_L5_MoreArff("weather.nominal")
	    .weight(5, 0.5);
	    
	    new W5_L5_MoreArff("weather.nominal")
	    .sparseFilter();
	}
	
	/*******************************************
	 * ��ǥ���� yes=firstWeight , no=secondWeight
	 *******************************************/
	public void weight(double firstWeight, double secondWeight) throws Exception{
		  
		for(Instance row : this.data){
			if(row.stringValue(row.numAttributes()-1).equals("yes"))
				row.setWeight(firstWeight);
			else
				row.setWeight(secondWeight);	
			
			System.out.println(row);
		}
	}
	

	/*******************************************
	 * �Ϲ����� arff ������ sparse arff ���Ϸ� ��ȯ
	 *******************************************/
	public void sparseFilter() throws Exception{

		System.out.println("�Ϲ� ����");	
		System.out.println(data);	
		
		NonSparseToSparse filter = new NonSparseToSparse();
		filter.setInputFormat(data);
		Instances sparsedata = Filter.useFilter(data, filter);	
		
		
		System.out.println("\nsparse ����");	
		System.out.println(sparsedata);
	}

	/****
	 * 
	 * CVParameterSelection ��  CVParameters ����
	 * 
	 ****/
	public void setCVParameters(CVParameterSelection optimizer, String optimizerOption) throws Exception{
		String[] options = new String[1];
		options[0] = optimizerOption;
		optimizer.setCVParameters(options);
	}
//	
//	private double printClassfiedInfo(Classifier model, Evaluation eval) throws Exception {
//		// 6) print Result text
//		System.out.print("\n** " + this.getModelName(model) +  
//                         ", �з���� ������ �� �� : " + (int)eval.numInstances() + 
//		                 ", ���з� �Ǽ� : " + (int)eval.correct() + 
//		                 ", ���з��� : " + String.format("%.1f",eval.pctCorrect()) +" %" +
//		                 ""); 	
//		System.out.print(", Weighted Area Under ROC : " + 
//        String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
//		System.out.println(eval.toMatrixString());
//		return 0;
//	}

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

}
