package _2_more;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.List;

import javax.swing.*;
import weka.associations.*;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class W3_L3_Association_Base{
	 Instances data=null;
	
	 public W3_L3_Association_Base(String fileName) throws Exception {
		 
		  // 1) data loader
		  String folderName = "D:\\Weka-3-9\\data\\vote\\";
		  System.out.print(fileName + " ");
		  this.data=new Instances(
		            new BufferedReader(
		            new FileReader(folderName+fileName+".arff")));
		  System.out.println("전체 데이터 건 수 : " + data.size());
	 }
	 
	 public static void main(String[] args) throws Exception {
		  W3_L3_Association_Base obj = new W3_L3_Association_Base();
		  
		  new W3_L3_Association_Base("vote_kor").association();
	 }
	 

	 // 연관규칙 실행
	 public void association() throws Exception{
		  // 2) class assigner 불필요
		  
		  // 3) evaluation 불필요, classifier setting
		  Apriori model = new Apriori(); // AbstractAssociator 상속된 하위 클래스
		  
		  // 4) model run 
		  model.buildAssociations(data);
		  
		  // 5) evaluate 불필요
		  
		  // 6) fetch the rules
		  AssociationRules rules = model.getAssociationRules();
		  
		  // 7) make & show rules	
		  List<AssociationRule> rule_list = rules.getRules();
		  
		  this.showRules(rule_list);
		  
		  System.out.println(model);
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
	 
	 public W3_L3_Association_Base(){
			try{
				Classifier model = new J48();
				model.buildClassifier(null);
			}catch(Exception e){
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
			}
	 }
}