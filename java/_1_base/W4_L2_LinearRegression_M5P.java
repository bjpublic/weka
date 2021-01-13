package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.*;

public class W4_L2_LinearRegression_M5P {

	public static void main(String args[]) throws Exception{
		W4_L2_LinearRegression_M5P obj = new W4_L2_LinearRegression_M5P();
		String fileName= "cpu";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression 실행을 위해
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ 접속하여
		 *  arpack_combined.jar, mtj.jar, core.jar 를 외부 jar 로 임포트 해야 한다.
		 ******************************************************************************/
		obj.cpuRegression(fileName,new LinearRegression());  
		
		// M5P는 위의 3개 jar 가 없어도 실행가능함.
		obj.cpuRegression(fileName,new M5P());  
	}

	public void cpuRegression(String fileName, Classifier model) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
//		Classifier model=classifier; // 매개변수에서 받은 생성된 model객체를 직접 사용		
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("model : " + model.toString() +"\n"+eval.toSummaryString()); // 회귀분석은 정분류율 보다 상관계수와 같은 회귀방정식 적정성 지표가 중요

		// 7) 학습된 모델로 추세 검증
		this.trend(model, data);
	}

	public void trend(Classifier model, Instances data) throws Exception{
		double differ = 0.0;
		double sumDifferABS = 0.0;
		double classValue = 0.0;
		double result = 0.0;
		for(int x=0 ; x < data.size() ; x++){
			Instance row = data.get(x);
			/**************************
			 * 생성된 모델로 class 계산 시작
			 **************************/
			result = model.classifyInstance(row);
			/**************************
			 * 생성된 모델로 class 계산 종료
			 **************************/
			classValue = row.valueSparse(row.numAttributes()-1);
			differ = result - classValue;
			sumDifferABS += Math.abs(differ); // 오차의 절대값 누적
			System.out.println( (x+1) + " : " + String.format("%.1f",classValue) + 
					                   " => " + String.format("%.1f",result) + 
					                   " 건별 오차 :" + String.format("%.1f",differ) );		
		}			
		System.out.println( "오차평균 : " + sumDifferABS/data.size());		
		
		if( model instanceof M5P){
			M5P m5p = (M5P)model; 
			// 다음기회에..
		}else{
			LinearRegression linear = (LinearRegression)model;
			// 다음기회에..
		}		
		
	}
}
