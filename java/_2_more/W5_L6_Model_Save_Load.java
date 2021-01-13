package _2_more;

import java.io.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.*;

public class W5_L6_Model_Save_Load {

	public static void main(String args[]) throws Exception{
		W5_L6_Model_Save_Load obj = new W5_L6_Model_Save_Load();
		obj.skip();
		System.out.println("=============================================================================");
		System.out.println("\t 1) BasePerceptron 에 분류기 대입 정분류 비교");
		System.out.println("=============================================================================");
				
	    Classifier[] models = {new J48(), new Logistic(), obj.getMultiPerceptron("a")}; 
	    
	    String[] fileNames = {"supermarket"};
	    for (String fileName : fileNames) {
	    	System.out.println("\nfileName : " + fileName);
		    for(Classifier model : models){
				obj.split(model,fileName, 66);
			}// end-for-models	
		}

		System.out.println("=============================================================================");
		System.out.println("\t 1) BasePerceptron");
		System.out.println("=============================================================================");
	}


	/*****************************
	 * split
	 *****************************/
	public void split(Classifier model, String fileName, int percent) throws Exception{
    	double startTime = System.currentTimeMillis();
		int seed = 1;
		// 1) data loader 
		String folderName = "D:\\Weka-3-9\\data\\";
		Instances data=new Instances(
				       new BufferedReader(
				       new FileReader(folderName+fileName+".arff")));
		
		/**********************************************************
		 * 1-1) 원본 데이터를 불러온 후 훈련/테스트 데이터로 분리 시작
		 **********************************************************/
		int trainSize = (int)Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		data.randomize(new java.util.Random(seed));
		
		Instances train = new Instances (data, 0 ,trainSize);
		Instances test  = new Instances (data, trainSize ,testSize);
		/**********************************************************
		 * 1-1) 원본 데이터를 불러온 후 훈련/테스트 데이터로 분리 종료
		 **********************************************************/
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
		// 3) holdout setting  
		Evaluation eval=new Evaluation(train);
		
		// 저장된 모델이 존재하는 지 확인
		if (this.loadModel(model) == null){
			// 훈련/테스트 데이터 분리되어 있으므로 교차검증 불필요
			
			// 4) model run 
			model.buildClassifier(train);
			
			// 모델 생성후 모델 저장
			this.saveModel(model);
			
			System.out.print(" SAVE MODEL : ");
		}else{	
			// 저장된 모델이 있으면 학습하지 않고 모델을 불러 옴
			model = (Classifier)this.loadModel(model);
			System.out.print(" LOAD MODEL : ");
		}	
		// 5) evaluate
		eval.evaluateModel(model, test);
			
		// 6) 정분류율, ROC
		this.printClassfiedInfo(model, eval, System.currentTimeMillis() - startTime);
		
	}	
	
	// 인공신경망 은닉층 설정
	public MultilayerPerceptron getMultiPerceptron(String hiddenlayers){
		
		MultilayerPerceptron perceptron = new MultilayerPerceptron();
		perceptron.setGUI(false); // true 로 변경하면 신경망 GUI 가 생성된다. 
		perceptron.setHiddenLayers(hiddenlayers);
		
		return perceptron;
	}
	
	private double printClassfiedInfo(Classifier model, Evaluation eval, double runTime) throws Exception {
		// 6) print Result text
		System.out.print(" 분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
		                 ", 정분류 건수 : " + (int)eval.correct() + 
		                 ", 정분류율 : " + String.format("%.1f",eval.pctCorrect()) +" %" +
		                 ""); 	
		System.out.print(", ROC : " + 
                         String.format("%.2f",eval.weightedAreaUnderROC()) + 
                         ", 실행시간 :" + String.format("%.2f",runTime/1000) +" sec " +
                         " ** " + this.getModelName(model) + "\n");
//		System.out.println(eval.toMatrixString());
		return 0;
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
			modelName = "MultilayerPerceptron" ;		
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

	// 모델 저장
	public void saveModel(Classifier model) throws Exception{
		String path = "C:\\Users\\bulle\\wekafiles\\model\\";
		weka.core.SerializationHelper.write(path +"supermarket_"+this.getModelName(model)+".model", model);
	}
	
	// 모델 불러오기
	public Object loadModel(Classifier model) throws Exception{
		if (this.getModelName(model).equals("MultilayerPerceptron")){
			// knowledgeFlow 에서 생성된 모델이 있는지 학인
			Object o = this.loadModel_from_KF(model); 
			if ( o != null ) {
				System.out.println(" FROM KF");
				return o; // knowledgeFlow 모델 반환
			}
		}
		
		String path = "C:\\Users\\bulle\\wekafiles\\model\\";
		try{
			return weka.core.SerializationHelper.read(path +"supermarket_"+this.getModelName(model)+".model");
		}catch (Exception e){
			return null;			
		}
	}

	// knowledgeFlow에서 생성된 모델 불러오기
	public Object loadModel_from_KF(Classifier model) throws Exception{
		String path = "C:\\Users\\bulle\\wekafiles\\model\\";
		try{
			return weka.core.SerializationHelper.read(path +"supermarket_1_1_MultilayerPerceptron.model");
		}catch (Exception e){
			return null;			
		}
	}
}	
