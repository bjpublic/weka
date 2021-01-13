package _1_base;

import java.io.*;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.OneR;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class W4_L3_classificationByRegressionFor2valueClass {

	public static void main(String args[]) throws Exception{
		W4_L3_classificationByRegressionFor2valueClass obj = new W4_L3_classificationByRegressionFor2valueClass();
		String fileName= "diabetes";
		System.out.println(fileName + " : ");
		
		/*****************************************************************************
		 *  LinearRegression 실행을 위해
		 *  https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/weka/lib/ 접속하여
		 *  arpack_combined.jar, mtj.jar, core.jar 를 외부 jar 로 임포트 해야 한다.
		 ******************************************************************************/
		// 1) 이산형으로 변형필터 적용한 데이터셋 반환 및 회귀식 생성 
		Instances filtterdData = obj.diabeteRegressionForNominalToBinaryFilter(fileName,new LinearRegression());

		// 2) AddClassification,  NumericToNominal, Remove 필터 적용
		filtterdData = obj.applyFilters(new LinearRegression(), filtterdData, "9", "1-8");
		
		// 3) OneR 분류에 의한 임계점 결정
//		obj.diabeteOneRForAddclassificationFilter(new OneR(),filtterdData,6);  // minBucketSize 적으면 과적합 발행
		obj.diabeteOneRForAddclassificationFilter(new OneR(),filtterdData,100);// 과적합 극복 위한 minBucketSize 확대  
	}

	public Instances diabeteRegressionForNominalToBinaryFilter(String fileName, Classifier model) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=new Instances(new BufferedReader(new FileReader("D:\\Weka-3-9\\data\\"+fileName+".arff")));
		/*****************************
		 * NominalToBinary 필터 적용 시작
		 *****************************/
		NominalToBinary filter = new NominalToBinary(); // unsupervised 를 선택 (supervised 에도 동일한 필터 존재
		filter.setAttributeIndices("last");
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		/*****************************
		 * NominalToBinary 필터 적용 종료
		 *****************************/

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
				
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		eval.crossValidateModel(model, train, numfolds, new Random(seed));
//		Classifier model=classifier; // 매개변수에서 받은 생성된 model객체를 직접 사용		
		
		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("\n**********************************************************************");
		System.out.println("\n         1) 이산형으로 변형필터 적용한 데이터셋 반환 및 회귀식 생성");
		System.out.println("\n**********************************************************************");
		System.out.println("1-1) NominalToBinary 적용후 data 속성개수 : " + data.numAttributes());
		System.out.println("1-2) 회귀식 model : " + model.toString() +"\n"+eval.toSummaryString()); 

		// 7) NominalToBinary 적용된 instances (데이터세트) 반환
		return data;
	}

	public void diabeteOneRForAddclassificationFilter(Classifier model, Instances filtterdData, int minBuckeSize) throws Exception{
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;		
		// 1) data loader 
		Instances data=filtterdData;
		data.setClassIndex(0);

		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner (필터링 과정에서 class 속성이 1번째(index=0)로 옮겨졌다)
		train.setClassIndex(0);
		test. setClassIndex(0);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		OneR classifier=(OneR)model;				
		/**********************************************************
		 * 과적합 방지를 위해 minBuckeSize 를 (매개변수로 받은) 100  으로 지정 시작
		 *********************************************************/
		classifier.setMinBucketSize(minBuckeSize);	
		/**********************************************************
		 * 과적합 방지를 위해 minBuckeSize 를 (매개변수로 받은) 100  으로 지정 종료
		 *********************************************************/
		
		// 4) model run 
		classifier.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(classifier, test);
		
		// 6) print Result text
		System.out.println("\n**********************************************************************");
		System.out.println("\n          3) OneR 분류에 의한 임계점 결정");
		System.out.println("\n**********************************************************************");
		System.out.println("3) minBuckeSize : "+ minBuckeSize + "\n classifier : " + classifier.toString() +"\n"+eval.toSummaryString()); // 회귀분석은 정분류율 보다 상관계수와 같은 회귀방정식 적정성 지표가 중요
	}
	
	/************************
	 * 각종 필터를 적용하기 위한 메소드
	 ************************/
	public Instances applyFilters(Classifier model, Instances data, String transIndicice, String removeIndices) throws Exception{
		System.out.println("\n**********************************************************************");
		System.out.println("\n          2) AddClassification,  NumericToNominal, Remove 필터 적용");
		System.out.println("\n**********************************************************************");
		data.setClassIndex(data.numAttributes()-1);
		// 1) AddClassification (숫자로만 된 예측결과 classification 속성 추가)
		AddClassification addfilter = new AddClassification();
		addfilter.setClassifier(model);
		addfilter.setOutputClassification(true);
		addfilter.setInputFormat(data);
		data = Filter.useFilter(data, addfilter);
		System.out.println("2-1) AddClassification 적용후 data 속성개수 : " + data.numAttributes());
		
		// 2) NumericToNominal (이산형으로 분리된 목적변수를 명목형으로 변환)
		NumericToNominal changeTypefilter = new NumericToNominal(); 
		changeTypefilter.setAttributeIndices(transIndicice);
		changeTypefilter.setInputFormat(data);
		data = Filter.useFilter(data, changeTypefilter);
		System.out.println("2-2) NumericToNominal 적용후 data 속성개수 : " + data.numAttributes());
		
		// 3) Remove (목적변수와 classification 만 남기고 모든속성 삭제)
		Remove filter = new Remove(); 
		filter.setAttributeIndices(removeIndices);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		System.out.println("2-3) Remove 적용후 data 속성개수 : " + data.numAttributes());
		
		return data;
	}

}
