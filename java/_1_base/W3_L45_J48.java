package _1_base;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.*;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.*;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class W3_L45_J48 {

	public static void main(String args[]) throws Exception{
		W3_L45_J48 obj = new W3_L45_J48();
		
		System.out.println("breast-cancer : ");
		obj.breastCancerJ48crossValidation(false);  
		obj.breastCancerJ48crossValidation(true);  
	}

	public void breastCancerJ48crossValidation(boolean isUnpruned) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\breast-cancer.arff")));
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		J48 model=new J48(); 		
		/*****************************
		 * unpruned (가지치기 않함) 설정 시작
		 *****************************/
		model.setUnpruned(isUnpruned);
		/*****************************
		 * unpruned (가지치기 않함) 설정 시작
		 *****************************/
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" 
				           + ", 분류기 : J48 , unprunded (가지치기 안함) : " + isUnpruned 
				           ); 	
		
		// 7) tree view
		this.treeVeiwInstances(data, model, eval);
		System.out.println(model);
	}

	 /**************************
	  * weka 제공 시각화 (treeView)
	  **************************/
	 public void treeVeiwInstances(Instances data, J48 tree, Evaluation eval) throws Exception {

		 String graphName = "";
		 graphName += " 정분류율 = " + String.format("%.2f",eval.pctCorrect()) + " %";
	     TreeVisualizer panel = new TreeVisualizer(null,tree.graph(),new PlaceNode2());
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     panel.fitToScreen();
	     System.out.println("See the " + graphName + " plot");
	 }     
	
}
