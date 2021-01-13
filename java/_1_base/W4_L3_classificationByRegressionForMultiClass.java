package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.OneR;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MakeIndicator;

public class W4_L3_classificationByRegressionForMultiClass {

	public static void main(String args[]) throws Exception{
		W4_L3_classificationByRegressionForMultiClass obj = new W4_L3_classificationByRegressionForMultiClass();
		String fileName= "iris";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression 실행을 위해
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ 접속하여
		 *  arpack_combined.jar, mtj.jar, core.jar 를 외부 jar 로 임포트 해야 한다.
		 ******************************************************************************/
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "last"); // versinica
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "2");    // versicolor
		obj.irisRegressionForMakeIndicatorFilter(fileName,new LinearRegression(), "1");    // setosa
 
	}

	public void irisRegressionForMakeIndicatorFilter(String fileName, Classifier model,String valueIndices) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;	
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		data.setClassIndex(data.numAttributes()-1); // 필터링을 위한 클래스 지정
		/*****************************
		 * MakeIndicator 필터 적용 시작
		 *****************************/
		MakeIndicator filter = new MakeIndicator(); 
		filter.setValueIndices(valueIndices);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		/*****************************
		 * MakeIndicator 필터 적용 종료
		 *****************************/
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
//		Classifier model=classifier; // 매개변수에서 받은 생성된 model객체를 직접 사용	
		
		// 4) model run 
		model.buildClassifier(data);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\n******************************************************");
		System.out.println("    model for " + filter.getValueIndices());
		System.out.println("******************************************************");
		System.out.println(model.toString() +"\n"+eval.toSummaryString()); // 회귀분석은 정분류율 보다 상관계수와 같은 회귀방정식 적정성 지표가 중요
		
		// 7) 임계점 산출
		System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 임계점 산출 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		this.makeCriticalPoint(data);
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 임계점 산출 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
		
		
	}
	
	public void makeCriticalPoint(Instances data) throws Exception{
		
		W4_L3_classificationByRegressionFor2valueClass obj = new W4_L3_classificationByRegressionFor2valueClass();


		// 2) AddClassification,  NumericToNominal, Remove 필터 적용
		data = obj.applyFilters(new LinearRegression(), data, "5", "1-4");
		
		// 3) OneR 분류에 의한 임계점 결정
		obj.diabeteOneRForAddclassificationFilter(new OneR(),data,100);// 과적합 극복 위한 minBucketSize 확대  
	}
}
