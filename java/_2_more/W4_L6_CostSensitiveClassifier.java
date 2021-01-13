package _2_more;

import java.io.*;
import java.util.Random;



import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W4_L6_CostSensitiveClassifier {
	Instances data=null;
	int[] selectedIdx = null; 
	
	public W4_L6_CostSensitiveClassifier (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		new W4_L6_CostSensitiveClassifier("credit-g")
			.classify(new NaiveBayes());

		new W4_L6_CostSensitiveClassifier("credit-g")
			.costClassifier(new NaiveBayes(),5,false);
		
		new W4_L6_CostSensitiveClassifier("credit-g")
			.costClassifier(new NaiveBayes(),5,true);

		new W4_L6_CostSensitiveClassifier("credit-g")
			.classify(new J48());

		new W4_L6_CostSensitiveClassifier("credit-g")
			.costClassifier(new J48(),5,false);

		new W4_L6_CostSensitiveClassifier("credit-g")
			.costClassifier(new J48(),5,true);
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
	
	public void costClassifier(Classifier model ,int FP_Weight, boolean minimizeExpectedCost ) throws Exception{		
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
		
		// label size
		int label_size = train.numClasses(); // matrix 사이즈 설정용
				
		// CostSensitiveClassifier 설정
		model = this.setCostSensitiveClassifier(model, label_size, 
				     this.setMatrix2X2(FP_Weight),minimizeExpectedCost);
		
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);		
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.print("\n** " + this.getModelName(model) +
				         ", minimizeExpectedCost = " + minimizeExpectedCost + " , " +
				         (minimizeExpectedCost?"(비용 분류,비용 최소화 분류)":"(비용 학습,내부적 가중치 재조정)") +
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
//        double TP = confusionMatrix[0][0];
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

		System.out.println("특이도 : " + 
				            TN + " / " + (FP+TN) + " = " + 
		                    String.format("%.2f", TN/(FP+TN) *100 ) + 
		                    " % ");
    }      
    	
    /****
     * 
     * CostSensitiveClassifier 설정
     * 
     ****/
	public Classifier setCostSensitiveClassifier(Classifier classifier, int size, int[][] matrix, boolean minimizeExpectedCost){
				
		// 분류 알고리즘 생성
		CostSensitiveClassifier costClassfier = new CostSensitiveClassifier();
		
		// set classifier
		costClassfier.setClassifier(classifier);	
		
		// set cost matrix 
		CostMatrix newCostMatrix = new CostMatrix(size);
				
		for(int x=0 ; x < matrix.length ; x++)
			for(int y=0; y < matrix[x].length ; y++)
				newCostMatrix.setElement(x, y, (double)matrix[x][y]);	
		
		costClassfier.setCostMatrix(newCostMatrix);
		
		// set minimizeExpectedCost
		costClassfier.setMinimizeExpectedCost(minimizeExpectedCost);
				
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

			System.out.println("cost matrix");
			System.out.println(costmatrix.toString()); // 설정된 cost matrix 출력
			
			matrix[0][1] = (int)Double.parseDouble(costmatrix.getCell(0, 1)+"");
			matrix[1][0] = (int)Double.parseDouble(costmatrix.getCell(1, 0)+"");
		}else{
			// 일반적인 알고리즘 기본값 설정
			matrix[0][1] = 1;
			matrix[1][0] = 1;
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
