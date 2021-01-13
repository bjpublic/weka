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
		  System.out.println("��ü ������ �� �� : " + data.size());
	 }
	 
	 public static void main(String[] args) throws Exception {
		  W3_L4_Association_More obj = new W3_L4_Association_More();
		  
		  new W3_L4_Association_More("vote_kor2").association();
	 }
	 

	 // ������Ģ ����
	 public void association() throws Exception{
		  // class assigner ���ʿ�
		  
		  // evaluation ���ʿ�, classifier setting
		  Apriori model = new Apriori(); // AbstractAssociator ��ӵ� ���� Ŭ����
		  
		  // model run 
		  model.buildAssociations(data);
		   
		  // evaluate ���ʿ�
		  
		  // fetch the rules
		  AssociationRules rules = model.getAssociationRules();
		  
		  // make & show rules	
		  List<AssociationRule> rule_list = rules.getRules();
		  
		  // ������Ģ �ĺ� �����ϰ� swing ���� ����ó�� ǥ��
		  this.showRules(rule_list);
		  
		  System.out.println(model);		  
		  
		  // 1) �������� A �� �������� B ���� �߻��� ��� �Ӽ����� �߻�ȸ�� ���
		  HashMap<String,Integer> attrNamesCounts = this.countByItemSets(rule_list);
		  
		  // �����ͼ�Ʈ �������� �Ӽ���� �Ӽ� index ����
		  ArrayList<String> attrNamesIndex = this.indexOfInstanceList(data);
		  
		  // 2) �ִ� �߻� index ����
		  int topIndex = this.fetchTopAttribute(attrNamesIndex, attrNamesCounts);

		  // 3) OneR �з��˰������� �ִ� �߻��Ӽ��� �����Ӽ� �� ������ Ȯ��
		  this.learnOneR(topIndex);
		  
	 }
	 
	 // 1) �������� A �� �������� B ���� �߻��� ��� �Ӽ����� �߻�ȸ�� ���
	 public HashMap<String,Integer> countByItemSets(List<AssociationRule> rule_list){

   	    HashMap<String,Integer> attrNamesCounts = new HashMap<String,Integer>();
		for (AssociationRule associationRule : rule_list) {
			
		  // �������� A
		  Collection<Item> premise = associationRule.getPremise();		  
		  attrNamesCounts = this.countByAttribute(premise, attrNamesCounts);

		  // �������� B
		  Collection<Item> consequence = associationRule.getConsequence();		  
		  attrNamesCounts = this.countByAttribute(consequence, attrNamesCounts);
  	    }	  
		
		return attrNamesCounts;
	 }
	 
	 // �Ӽ��� �߻�ȸ�� ����
	 public HashMap<String,Integer> countByAttribute(Collection<Item> itemSet, HashMap<String,Integer> attrNamesCounts){

		  for (Iterator<Item> iterator2 = itemSet.iterator(); iterator2.hasNext();) {
			  Item itemPremise = (Item) iterator2.next();
			  
			  // �Ӽ���
			  String attrName = itemPremise.getAttribute().name();
			  
			  // �Ӽ��� �߻�ȸ�� ����
			  if(attrNamesCounts.get(attrName) != null){
				 int count = Integer.parseInt(attrNamesCounts.get(attrName)+"") + 1;
				 attrNamesCounts.put(attrName, Integer.valueOf(count));
			  }else{
				 attrNamesCounts.put(attrName, Integer.valueOf(1)); 
			  }
		  }
		  
		  return attrNamesCounts;
	 }
	 
	 // �ִ� �߻� index ����
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
		 System.out.println("�ִ� �߻� �Ӽ� index =" + (topIndex+1) + " , " + topAttrName + " = " + topCount);
		 
		 return topIndex;
	 }

	 // �����ͼ�Ʈ �������� �Ӽ���� �Ӽ� index ����
	 public ArrayList<String> indexOfInstanceList(Instances data){
		ArrayList<String> attrNamesIndex = new ArrayList<String>();
		Instance instance = data.firstInstance();
		for (int x=0 ; x < instance.numAttributes() ; x++){
			Attribute attr = instance.attribute(x);
			attrNamesIndex.add(attr.name());
		}
		return attrNamesIndex;
	 }
	 
	 // OneR �� �з���Ģ�� ���з��� (������) ����
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
		System.out.println("\t�з���� ������ �� �� : " + (int)eval.numInstances() + 
				           ", ���з� �Ǽ� : " + (int)eval.correct() + 
				           ", ���з��� : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %"+ 
				           ", seed : " + seed ); 	
		// 7) �з���Ȯ�� ��ȯ
//		return eval.correct() / eval.numInstances() * 100;
		 
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
	 
	 public W3_L4_Association_More(){
			try{
				Classifier model = new J48();
				model.buildClassifier(null);
			}catch(Exception e){
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
			}
	 }
}