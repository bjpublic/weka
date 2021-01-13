package _2_more;

import java.awt.*;
import java.io.*;
import java.util.HashMap;
import java.util.Random;

import javax.swing.*;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.clusterers.Clusterer;
import weka.clusterers.Cobweb;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class W4_L2_AttrSelClassifier {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L2_AttrSelClassifier (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		new W4_L2_AttrSelClassifier("glass")
		   .loop(new J48(),new NaiveBayes());
		

		new W4_L2_AttrSelClassifier("glass")
		   .loop(new NaiveBayes(),new J48());
	}

	public void loop(Classifier model, Classifier classifier) throws Exception{		
		System.out.println("****************************** " + 
	                       this.getModelName(model)+ 
	                       " ******************************");
		for(int x=0 ; x <4 ; x++){
			this.attrSelClassifier(x, model, classifier);
		}
	}
	
	public void attrSelClassifier(int testCase, Classifier model, Classifier classifier) throws Exception{		
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		
		Instances attrSelectInstances = null;
		
		switch(testCase){
			case 0:
				// 1) wrapper and model cross (good)
				System.out.println("\t ================== 1) 역주행 (good) :");
				attrSelectInstances = this.wrappersubset(classifier);
				break;
			case 1:
				// 1) wrapper and model go (bad)
				System.out.println("\t ================== 2) 정주행 (bad) :");
				attrSelectInstances = this.wrappersubset(model);
				break;
			case 2:
				System.out.println("\t ================== 3) attributeSelectedClassifer 실행 (bad) :");
				model = this.attrSelClassifier(model);
				attrSelectInstances = this.data;
				break;
			case 3:
				System.out.println("\t ================== 4) 일반적인 경우 실행 (good) :");
				attrSelectInstances = this.data;
				break;
		}
		
		// 1) data split 
		Instances train = attrSelectInstances.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = attrSelectInstances.testCV (numfolds, numfold);

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
		System.out.println("\t\t\t" + (testCase+1) + " 번째, " + 
		                   "분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
				           "\n"); 	
		
		// 7) tree view
//		this.treeVeiwInstances(testCase + " 번째 트리, 정분류율 : " 
//		                     + String.format("%.1f",eval.pctCorrect()) + " %" , model);
	}
	
	/**
	 * Wrappersubset
	 * */
	public Instances wrappersubset(Classifier classifier) throws Exception{
		
		// "select attribute" 패널 역할 객체
		AttributeSelection attrSelector = new AttributeSelection();
		
		Instances wrapped_Instance = this.data;		
		wrapped_Instance.setClassIndex(wrapped_Instance.numAttributes()-1);
		
		// BestFirst default
		
		// WrapperSubsetEval
		WrapperSubsetEval subset = new WrapperSubsetEval();
		subset.setClassifier(classifier);

		// BestFirst, WrapperSubsetEval setting
		attrSelector.setEvaluator(subset);
		
		// RUN Selection of Attribute
		attrSelector.SelectAttributes(wrapped_Instance);
		
		// get selected index count
		this.selectedIdx = attrSelector.selectedAttributes();
		
		// print index of selected attributes
		for (int x=0 ; x < this.selectedIdx.length ; x++) {			
			// get selected attribute name
			Attribute attr = attrSelector
					        .reduceDimensionality(wrapped_Instance)
					        .attribute(x);
			String attrName= attr.name();
			System.out.println("\t\t selected attribute index : " 
			                   + (this.selectedIdx[x]+1) 
			                   + " : " + attrName);
		}
		
		// return selected data (reduced attribute)
		return attrSelector.reduceDimensionality(wrapped_Instance);
	}
	
	public Classifier attrSelClassifier(Classifier classifier){
				
		// 분류 알고리즘 생성
		AttributeSelectedClassifier attrSelClassifier = null;
		
		// 상위 classifier 설정
		attrSelClassifier = new AttributeSelectedClassifier();
		attrSelClassifier.setClassifier(classifier);
		
		// evaluator 의 classifier 설정
		WrapperSubsetEval subset = new WrapperSubsetEval();
		subset.setClassifier(classifier);
		attrSelClassifier.setEvaluator(subset);
				
		return attrSelClassifier;
	}

	public String getModelName(Classifier Classifier) throws Exception{
		String modelName = "";
		if(Classifier instanceof J48){
			modelName = "J48";
		}else if(Classifier instanceof NaiveBayes){
			modelName = "NaiveBayes";		
		}
		
		return modelName;
	}
	
	 /**************************
	  * 트리 출력
	  **************************/
	 public void treeVeiwInstances(String treeType, J48 model) throws Exception {

		 String graphName = treeType;
	     TreeVisualizer panel = new TreeVisualizer(null,
	    		 					model.graph(),
	    		                    new PlaceNode2());
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     panel.fitToScreen();
	 }     

}
