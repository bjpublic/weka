package _2_more;

import java.io.*;
import java.util.Random;



import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L5_CostMatrix {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L5_CostMatrix (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		new W4_L5_CostMatrix("breast-cancer")
			.classify(new NaiveBayes());

		new W4_L5_CostMatrix("breast-cancer")
			.costClassifier(new NaiveBayes(),2,2);

		new W4_L5_CostMatrix("breast-cancer")
			.classify(new J48());
		
		new W4_L5_CostMatrix("breast-cancer")
		.costClassifier(new J48(),2,2);

		new W4_L5_CostMatrix("breast-cancer")
		.costClassifier(new J48(),2,5);

		new W4_L5_CostMatrix("breast-cancer")
		.costClassifier(new ZeroR(),2,5);
	}
	
	public void classify(Classifier model) throws Exception{	
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
				
		// 1) data split 
		Instances train = this.data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = this.data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test.numAttributes()-1);
		
		// 3) eval and model setting  
		Evaluation eval=new Evaluation(train);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);		
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.print("\n** " + this.getModelName(model) +  
		                   ", 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
				           ""); 	
		System.out.print(", Weighted Area Under ROC : " + 
                           String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
		System.out.println(eval.toMatrixString());
		this.printPctCorrect2ndLabel(eval.confusionMatrix(), this.getMatrix2X2(null));	
		
	}
	
	public void costClassifier(Classifier model , int size, int FP_Weight) throws Exception{		
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
				
		// 1) data split 
		Instances train = this.data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = this.data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(train.numAttributes()-1);
		
		// 3) eval and model setting  
		Evaluation eval=new Evaluation(train);
				
		model = this.costMatrix(model, size, this.setMatrix2X2(FP_Weight));
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);		
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.print("\n** " + this.getModelName(model) +  
                         ", 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		                 ", 정분류 건수 : " + (int)eval.correct() + 
		                 ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
		                 ""); 	
		System.out.print(", Weighted Area Under ROC : " + 
        String.format("%.2f",eval.weightedAreaUnderROC()) + "\n");
		System.out.println(eval.toMatrixString());
		this.printPctCorrect2ndLabel(eval.confusionMatrix(), 
				                     this.getMatrix2X2((CostSensitiveClassifier)model));
	}

    // 분류결과 출력
    public void printPctCorrect2ndLabel(double[][] confusionMatrix, int[][] matrix){
        double TP = confusionMatrix[0][0];
        double FN = confusionMatrix[0][1];
        double FP = confusionMatrix[1][0];
        double TN = confusionMatrix[1][1];
        System.out.println("total cost : " + (int)(matrix[1][0] * FP + matrix[0][1] * FN)
        								   + "="+matrix[1][0]+"*"+(int)FP 
        								   + "+"+matrix[0][1]+"*"+(int)FN);
        
        System.out.println("total miss : " + (int)(FP+FN) + "="
                                           + (int)FN +"+"
        		                           + (int)FP);
        
        System.out.println("critical miss : " + (int)FP);
    }      
    	
	public Classifier costMatrix(Classifier classifier, int size, int[][] matrix){
				
		// 분류 알고리즘 생성
		CostSensitiveClassifier costClassfier = null;
		
		CostMatrix newCostMatrix = null;
		
		newCostMatrix = new CostMatrix(size);
				
		for(int x=0 ; x < matrix.length ; x++)
			for(int y=0; y < matrix[x].length ; y++)
				newCostMatrix.setElement(x, y, (double)matrix[x][y]);
		
		// 상위 classifier 설정
		costClassfier = new CostSensitiveClassifier();
		costClassfier.setClassifier(classifier);
		
		costClassfier.setCostMatrix(newCostMatrix);
				
		return costClassfier;
	}
	
	public int[][] setMatrix2X2(int FP_Weight){
		int[][] matrix = new int [2][2]; 
		matrix[1][0] = FP_Weight;
		matrix[0][1] = 1;
		
		return matrix;
	}
	
	public int[][] getMatrix2X2(CostSensitiveClassifier model){
		int[][] matrix = new int [2][2]; 
		if(model != null){
			CostMatrix costmatrix = model.getCostMatrix();

			System.out.println("cost matrxi");
			System.out.println(costmatrix.toString());
			
			matrix[0][1] = (int)Double.parseDouble(costmatrix.getCell(0, 1)+"");
			matrix[1][0] = (int)Double.parseDouble(costmatrix.getCell(1, 0)+"");
		}
		
		return matrix;
	}
	
	public String getModelName(Classifier Classifier) throws Exception{
		String modelName = "";
		if(Classifier instanceof NaiveBayesMultinomial){
			modelName = "NaiveBayesMultinomial";
		}else if(Classifier instanceof NaiveBayes){
			modelName = "NaiveBayes";		
		}else if(Classifier instanceof J48){
			modelName = "J48";		
		}else if(Classifier instanceof CostSensitiveClassifier){
			modelName = "CostSensitiveClassifier";		
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
