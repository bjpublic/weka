package _2_more;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Hashtable;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class W1_L2_TTest_Classifiers {
	
	Instances data = null;
	Hashtable<String, Double> map = null;
	double[] sample1 = new double[100];
	double[] sample2 = new double[100];
	
	public static void main(String[] args) throws Exception{
		W1_L2_TTest_Classifiers test = new W1_L2_TTest_Classifiers();
//		test.sample1 = {43, 21, 25, 42, 57, 59, 25, 42, 57, 59};
//		test.sample2 = {99, 65, 79, 75, 87, 81, 25, 42, 57, 59};
//		test.getTtest(sample1, sample2);
		
		W1_L2_TTest_Classifiers obj = new W1_L2_TTest_Classifiers();
		
		obj.getDataSet();
		
		obj.saveRelations("breast-cancer", "weka.classifiers.trees.J48", "weka.classifiers.functions.SMO");
		
		obj.aggregateValue(obj.sample1);

		obj.aggregateValue(obj.sample2);
		
		obj.getTtest();
		
		System.out.println("end");
	}
	
	public void getDataSet() throws Exception{

		this.data=new Instances(
			      new BufferedReader(
			      new FileReader("D:\\Weka-3-9\\experimenter\\more 1.2.arff")));
	}
	
	public void saveRelations(String dataSetName, String classifier1_Name, String classifier2_Name ){
		
		int sample1_index = 0;
		int sample2_index = 0;
		for(int x=0 ; x < this.data.numInstances() ; x++){
			Instance row = this.data.get(x);
			String dataSet = row.stringValue(0); // dataset name
			String classifier = row.stringValue(3); // classifier name
			double pctCorrect = row.value(12); // correct ratio
			
			if (dataSet.equals(dataSetName)){
				if(classifier.equals(classifier1_Name)){
					this.sample1[sample1_index++] = pctCorrect;
					System.out.println(sample1_index + " , dataSet = " + dataSet + " , classifier = " + classifier + " , " + pctCorrect );
				}else if (classifier.equals(classifier2_Name)){
					this.sample2[sample2_index++] = pctCorrect;
					System.out.println(sample2_index + " , dataSet = " + dataSet + " , classifier = " + classifier + " , " + pctCorrect );
				}	
			}
			
			if (sample1_index == 100) sample1_index = 0;
			if (sample2_index == 100) sample2_index = 0;
		}
	}
	
	public void getTtest(){
		System.out.println(TestUtils.pairedT(sample1, sample2));//t statistics
		System.out.println(TestUtils.pairedTTest(sample1, sample2));//p value
		System.out.println(TestUtils.pairedTTest(sample1, sample2, 0.05));

		System.out.println(TestUtils.pairedT(sample2, sample1));//t statistics
		System.out.println(TestUtils.pairedTTest(sample2, sample1));//p value
		System.out.println(TestUtils.pairedTTest(sample2, sample1, 0.05));
		
		System.out.println(TestUtils.tTest(sample2, sample1));		
		System.out.println(TestUtils.tTest(sample1, sample2));
	}
	
	public void aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  sumObj.addValue(sum[i]); 

		System.out.println("Æò±Õ : " + String.format("%.2f",aggregate.getMean()) + " %, ÆíÂ÷ : " + String.format("%.2f",aggregate.getStandardDeviation()) + " %");
	}
}
