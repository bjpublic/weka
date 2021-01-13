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
		  System.out.println("��ü ������ �� �� : " + data.size());
	 }
	 
	 public static void main(String[] args) throws Exception {
		  W3_L3_Association_Base obj = new W3_L3_Association_Base();
		  
		  new W3_L3_Association_Base("vote_kor").association();
	 }
	 

	 // ������Ģ ����
	 public void association() throws Exception{
		  // 2) class assigner ���ʿ�
		  
		  // 3) evaluation ���ʿ�, classifier setting
		  Apriori model = new Apriori(); // AbstractAssociator ��ӵ� ���� Ŭ����
		  
		  // 4) model run 
		  model.buildAssociations(data);
		  
		  // 5) evaluate ���ʿ�
		  
		  // 6) fetch the rules
		  AssociationRules rules = model.getAssociationRules();
		  
		  // 7) make & show rules	
		  List<AssociationRule> rule_list = rules.getRules();
		  
		  this.showRules(rule_list);
		  
		  System.out.println(model);
	 }
	 
	// swing ���̺� ���� ����
	 public void showRules(List<AssociationRule> rule_list) throws Exception{
		  String header[] = {"����","source","��������A (lhs)","�������� B (rhs)",
		                     "�ŷڵ� (confidence)","��� (lift)",
				             "�������� A ������ (weka)","�������� B ������ (weka)","��ü ������"};
		  String contents[][] = new String[rule_list.size()][9];
		  StringBuffer sbuffer = null;
		  int i=0;

		  for (AssociationRule associationRule : rule_list) {
			  contents[i][0] = (i+1)+"";
			  contents[i][1] = "Weka";

			  // �������� A 
			  Collection<Item> premise = associationRule.getPremise(); 
			  sbuffer = new StringBuffer();
			  for (Iterator<Item> iterator2 = premise.iterator(); iterator2.hasNext();) {
				  Item itemPremise = (Item) iterator2.next();

				  sbuffer.append( itemPremise.getAttribute().name() + "=" + // �������� A �Ӽ���	  
		                          itemPremise.getItemValueAsString() +", " );  // �������� A �Ӽ���	 
			  }
			  contents[i][2] = sbuffer.toString();

			  // �������� B
			  Collection<Item> consequence = associationRule.getConsequence(); 
			  sbuffer = new StringBuffer();
			  for (Iterator<Item> iterator = consequence.iterator(); iterator.hasNext();) {
				Item itemConsequence = (Item) iterator.next();

				sbuffer.append( itemConsequence.getAttribute().name() + "=" + // �������� B �Ӽ���	  
				       		    itemConsequence.getItemValueAsString() +", ");  // �������� B �Ӽ���	
			  }
			  
			  contents[i][3] = sbuffer.toString();
			  
			  double matric[] = associationRule.getMetricValuesForRule();
			  contents[i][4] = String.format("%.2f",matric[0]) + "";  // �ŷڵ�
			  contents[i][5] = String.format("%.2f",matric[1]) + "";  // ���			  		  
			  
			  contents[i][6] = associationRule.getPremiseSupport()+""; // �������� A ������ (weka)
			  contents[i][7] = associationRule.getTotalSupport()+"";   // �������� B ������ (weka)
			  contents[i][8] = associationRule.getConsequenceSupport()+"";         // ��ü ������

			  i++;
		  }		 
		  
		  this.makeTable(header, contents);
	 }
	 
	 
	 // swing ���̺�� ������Ģ ���
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