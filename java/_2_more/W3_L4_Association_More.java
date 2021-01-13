package _2_more;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.List;

import javax.swing.*;
import weka.associations.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.*;

public class W3_L4_Association_More{
	 Instances data=null;
	
	 public W3_L4_Association_More(String fileName) throws Exception {
		 
		  // 1) data loader
		  String folderName = "D:\\Weka-3-9\\data\\vote\\";
		  System.out.print(fileName + " ");
		  this.data=new Instances(
		            new BufferedReader(
		            new FileReader(folderName+fileName+".arff")));
		  System.out.println("전체 데이터 건 수 : " + data.size());
	 }
	 
	 public static void main(String[] args) throws Exception {
		  W3_L4_Association_More obj = new W3_L4_Association_More();
		  
		  new W3_L4_Association_More("vote_kor2").association();
	 }
	 

	 // 연관규칙 실행
	 public void association() throws Exception{
		  // class assigner 불필요
		  
		  // evaluation 불필요, classifier setting
		  Apriori model = new Apriori(); // AbstractAssociator 상속된 하위 클래스
		  
		  // model run 
		  model.buildAssociations(data);
		   
		  // evaluate 불필요
		  
		  // fetch the rules
		  AssociationRules rules = model.getAssociationRules();
		  
		  // make & show rules	
		  List<AssociationRule> rule_list = rules.getRules();
		  
		  // 연관규칙 식별 용이하게 swing 으로 엑셀처럼 표현
		  this.showRules(rule_list);
		  
		  System.out.println(model);		  
		  
		  // 1) 전조현상 A 와 병행현상 B 에서 발생한 모든 속성값별 발생회수 계산
		  HashMap<String,Integer> attrNamesCounts = this.countByItemSets(rule_list);
		  
		  // 데이터세트 구조에서 속성명과 속성 index 저장
		  ArrayList<String> attrNamesIndex = this.indexOfInstanceList(data);
		  
		  // 2) 최다 발생 index 지정
		  int topIndex = this.fetchTopAttribute(attrNamesIndex, attrNamesCounts);

		  // 3) OneR 분류알고리즘으로 최다 발생속성과 연관속성 및 밀접도 확인
		  this.learnOneR(topIndex);
		  
	 }
	 
	 // 1) 전조현상 A 와 병행현상 B 에서 발생한 모든 속성값별 발생회수 계산
	 public HashMap<String,Integer> countByItemSets(List<AssociationRule> rule_list){

   	    HashMap<String,Integer> attrNamesCounts = new HashMap<String,Integer>();
		for (AssociationRule associationRule : rule_list) {
			
		  // 전조현상 A
		  Collection<Item> premise = associationRule.getPremise();		  
		  attrNamesCounts = this.countByAttribute(premise, attrNamesCounts);

		  // 병행현상 B
		  Collection<Item> consequence = associationRule.getConsequence();		  
		  attrNamesCounts = this.countByAttribute(consequence, attrNamesCounts);
  	    }	  
		
		return attrNamesCounts;
	 }
	 
	 // 속성명별 발생회수 저장
	 public HashMap<String,Integer> countByAttribute(Collection<Item> itemSet, HashMap<String,Integer> attrNamesCounts){

		  for (Iterator<Item> iterator2 = itemSet.iterator(); iterator2.hasNext();) {
			  Item itemPremise = (Item) iterator2.next();
			  
			  // 속성명
			  String attrName = itemPremise.getAttribute().name();
			  
			  // 속성명 발생회수 저장
			  if(attrNamesCounts.get(attrName) != null){
				 int count = Integer.parseInt(attrNamesCounts.get(attrName)+"") + 1;
				 attrNamesCounts.put(attrName, Integer.valueOf(count));
			  }else{
				 attrNamesCounts.put(attrName, Integer.valueOf(1)); 
			  }
		  }
		  
		  return attrNamesCounts;
	 }
	 
	 // 최다 발생 index 지정
	 public int fetchTopAttribute(ArrayList<String> attrNamesIndex, HashMap<String,Integer> attrNamesCounts){
		 String topAttrName = "";
		 int topCount = 0;
		 int topIndex = 0;
		 for (int x=0 ; x < attrNamesIndex.size()-1 ; x++){
			 String currAttrName = attrNamesIndex.get(x)+"";
			 if( currAttrName != null ){
				 int currCount = 0;
				 try{
					 currCount = Integer.parseInt(attrNamesCounts.get(currAttrName)+"");
				 }catch(Exception e){
					 
				 }
				 if(currCount > topCount){
					 topCount = currCount;
					 topAttrName = currAttrName;
					 topIndex = x;
				 }
			 }	 
		 }
		 System.out.println("최다 발생 속성 index =" + (topIndex+1) + " , " + topAttrName + " = " + topCount);
		 
		 return topIndex;
	 }

	 // 데이터세트 구조에서 속성명과 속성 index 저장
	 public ArrayList<String> indexOfInstanceList(Instances data){
		ArrayList<String> attrNamesIndex = new ArrayList<String>();
		Instance instance = data.firstInstance();
		for (int x=0 ; x < instance.numAttributes() ; x++){
			Attribute attr = instance.attribute(x);
			attrNamesIndex.add(attr.name());
		}
		return attrNamesIndex;
	 }
	 
	 // OneR 로 분류규칙과 정분류율 (밀접도) 측정
	 public void learnOneR(int topIndex) throws Exception{
		 
		 this.data.setClassIndex(topIndex);
		 System.out.println(this.data.classIndex() + " , " + this.data.classAttribute());
		 
		int seed = 0;
		int numfolds = 10;
		int numfold = 0;
		
		// 1) data split
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(topIndex);
		test.setClassIndex(topIndex);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		OneR model = new OneR();
		eval.crossValidateModel(model, train, numfolds, new Random(seed)); 

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		System.out.println(model);
		
		// 6) print Result text
		System.out.println("\t분류대상 데이터 건 수 : " + (int)eval.numInstances() + 
				           ", 정분류 건수 : " + (int)eval.correct() + 
				           ", 정분류율 : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
				           ", seed : " + seed ); 	
		// 7) 분류정확도 반환
//		return eval.correct() / eval.numInstances() * 100;
		 
	 }
	 

	// swing 테이블 내용 생성
	 public void showRules(List<AssociationRule> rule_list) throws Exception{
		  String header[] = {"순서","source","전조현상A (lhs)","변행현상 B (rhs)",
		                     "신뢰도 (confidence)","향상도 (lift)",
				             "전조현상 A 지지도 (weka)","병행현상 B 지지도 (weka)","전체 지지도"};
		  String contents[][] = new String[rule_list.size()][9];
		  StringBuffer sbuffer = null;
		  int i=0;

		  for (AssociationRule associationRule : rule_list) {
			  contents[i][0] = (i+1)+"";
			  contents[i][1] = "Weka";

			  // 전조현상 A 
			  Collection<Item> premise = associationRule.getPremise(); 
			  sbuffer = new StringBuffer();
			  for (Iterator<Item> iterator2 = premise.iterator(); iterator2.hasNext();) {
				  Item itemPremise = (Item) iterator2.next();

				  sbuffer.append( itemPremise.getAttribute().name() + "=" + // 전조현상 A 속성명	  
		                          itemPremise.getItemValueAsString() +", " );  // 전조현상 A 속성값	 
			  }
			  contents[i][2] = sbuffer.toString();

			  // 변행현상 B
			  Collection<Item> consequence = associationRule.getConsequence(); 
			  sbuffer = new StringBuffer();
			  for (Iterator<Item> iterator = consequence.iterator(); iterator.hasNext();) {
				Item itemConsequence = (Item) iterator.next();

				sbuffer.append( itemConsequence.getAttribute().name() + "=" + // 병행현상 B 속성명	  
				       		    itemConsequence.getItemValueAsString() +", ");  // 병행현상 B 속성값	
			  }
			  
			  contents[i][3] = sbuffer.toString();
			  
			  double matric[] = associationRule.getMetricValuesForRule();
			  contents[i][4] = String.format("%.2f",matric[0]) + "";  // 신뢰도
			  contents[i][5] = String.format("%.2f",matric[1]) + "";  // 향상도			  		  
			  
			  contents[i][6] = associationRule.getPremiseSupport()+""; // 전조현상 A 지지도 (weka)
			  contents[i][7] = associationRule.getTotalSupport()+"";   // 병행현상 B 지지도 (weka)
			  contents[i][8] = associationRule.getConsequenceSupport()+"";         // 전체 지지도

			  i++;
		  }		 
		  
		  this.makeTable(header, contents);
	 }
	 
	 
	 // swing 테이블로 연관규칙 출력
	 public void makeTable(String[] header, String[][] contents){
		 Dimension dim = new Dimension(1500,250);
		 JFrame frame = new JFrame("Apriori rules table");
		 frame.setLocation(10, 10);
		 frame.setPreferredSize(dim);
		 
		 JTable table = new JTable(contents, header);
		 JScrollPane scrollpane = new JScrollPane(table);
		 frame.add(scrollpane);
		 frame.pack();
		 frame.setVisible(true);
	 }
	 
	 public W3_L4_Association_More(){
			try{
				Classifier model = new J48();
				model.buildClassifier(null);
			}catch(Exception e){
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
			}
	 }
}