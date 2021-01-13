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
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class W4_L1_Wrapper {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L1_Wrapper (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		Classifier model = new IBk(); 
		new W4_L1_Wrapper("glass").wrapper(1,model);
		new W4_L1_Wrapper("glass").wrapper(2,model);
		new W4_L1_Wrapper("glass").wrapper(3,model);
	}

	public void wrapper(int testCase, Classifier model) throws Exception{		
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		
		Instances wrapped_Instance = null;
		switch(testCase){
			case 1 : // 그대로 학습
				wrapped_Instance = this.data; 
				break;
			case 2 : // 속성선택
				wrapped_Instance = this.wrappersubset(1, -1.0, model); 
				break;
			case 3 : // remove 필터링 적용
				this.wrappersubset(1, -1.0, model); 
				int[] unSel = this.unSelectedAttributes();
				String str = "";
				for(int x=0 ; x < unSel.length ; x++) str += (unSel[x]+1) + ",";
				wrapped_Instance = this.removeFilter(str); 
//				wrapped_Instance = this.removeFilter("2,5,7,9"); 
				break;
		}
		
		// 1) data split 
		Instances train = wrapped_Instance.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = wrapped_Instance.testCV (numfolds, numfold);

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
		System.out.println("\t" + testCase + " 번째, " + 
		                   "분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" 
				           ); 	
		
		// 7) tree view
//		this.treeVeiwInstances(testCase + " 번째 트리, 정분류율 : " 
//		                     + String.format("%.1f",eval.pctCorrect()) + " %" , model);
	}
	
	/**
	 * wrappersubset 속성선택
	 * */
	public Instances wrappersubset(int t, double theshold, Classifier classifier) throws Exception{
		
		// "select attribute" 패널 역할 객체
		AttributeSelection attrSelector = new AttributeSelection();
		
		Instances wrapped_Instance = this.data;		
		wrapped_Instance.setClassIndex(wrapped_Instance.numAttributes()-1);
		
		// BestFirst
		BestFirst first = new BestFirst();		
		first.setSearchTermination(t);
		
		// WrapperSubsetEval
		WrapperSubsetEval subset = new WrapperSubsetEval();
		subset.setClassifier(classifier);
		subset.setThreshold(theshold);

		// BestFirst, WrapperSubsetEval setting
		attrSelector.setEvaluator(subset);
		attrSelector.setSearch(first);
		
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
			System.out.println("selected attribute index : " 
			                   + (this.selectedIdx[x]+1) 
			                   + " : " + attrName);
		}
		
		// return selected data (reduced attribute)
		return attrSelector.reduceDimensionality(wrapped_Instance);
	}
	
	
	// 선택 안된 속성값 추출 (자바 데이터 구조 기준 : 0부터 시작)
	public int[] unSelectedAttributes(){
		
		HashMap<Integer, String> selectedIdxMap = new HashMap<Integer, String>();
		for (int y=0 ; y < this.selectedIdx.length ; y++)
			selectedIdxMap.put(new Integer(this.selectedIdx[y]), "");
		
		int size = this.data.numAttributes() - this.selectedIdx.length;				
		int[] unSelAttr = new int [size];

		int xx=0;
		for(int x=0 ; x < this.data.numAttributes() ; x++){
			if(selectedIdxMap.get(Integer.parseInt(x+"")) == null){
//				System.out.println(Integer.parseInt(x+""));
				unSelAttr[xx++] = Integer.parseInt(x+"");
			}
		}
		return unSelAttr;
	}
	
	/**
	 * remove 필터링
	 * */
	public Instances removeFilter(String indexes) throws Exception{
		Remove filter = new Remove();
		filter.setAttributeIndices(indexes);
		filter.setInputFormat(data);
		System.out.println("removed index : " 
		                   +  filter.getAttributeIndices());
		return Filter.useFilter(data, filter);
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
