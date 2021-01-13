package _2_more;

import java.awt.*;
import java.io.*;
import java.util.Random;

import javax.swing.*;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.clusterers.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class W3_L6_Cluster_Eval {
	Instances data=null;
	ClusterEvaluation eval=null;
	
	public W3_L6_Cluster_Eval (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		new W3_L6_Cluster_Eval("iris").addCluster(new SimpleKMeans(), 3);
		new W3_L6_Cluster_Eval("iris").eval_Cluster(new SimpleKMeans(), 3);
	}

	public void addCluster(SimpleKMeans cluster, double k) throws Exception{		
		// 2) class assigner ���ʿ� , ��� class �� ����
		AddCluster filter = new AddCluster();
		cluster.setNumClusters((int)k);
		filter.setClusterer(cluster);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);	
		
		System.out.println("=================================================");
		System.out.println("\t 1) �ð�ȭ");
		System.out.println("=================================================");
		this.dataVeiwInstances((int)k);
		System.out.println("=================================================");
		System.out.println("\t 2) Addcluster ���͸� ���");
		System.out.println("=================================================");
		this.makeTable(cluster);
	}

	public void eval_Cluster(SimpleKMeans cluster, double k) throws Exception{	
		int seed = 1;
		int numfolds = 10;
		int numfold = 0;
		
		// 1) data loader 
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);

		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test. setClassIndex(test. numAttributes()-1);
		
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);
		
	    ClassificationViaClustering model=new ClassificationViaClustering(); 	
	    SimpleKMeans kmeans = new SimpleKMeans();
	    kmeans.setNumClusters((int)k);
	    model.setClusterer(kmeans);
	    
		eval.crossValidateModel(model, train, numfolds, new Random(seed));

		// 4) model run 
		model.buildClassifier(train);
		
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text
		System.out.println("=================================================");
		System.out.println("\t 4) �з��� ���з��� �� ����������� ���");
		System.out.println("=================================================");
		System.out.println("���з��� : " + String.format("%.2f",eval.pctCorrect()) + " %");
		System.out.println("������ �Ҵ� �� �� : ");
		System.out.println(model);
		System.out.println("\n " + eval.toMatrixString());
		
	}
	
	public String getClusterModelName(Clusterer cluster) throws Exception{
		String modelName = "";
		if(cluster instanceof SimpleKMeans){
			modelName = "Kmeasns";
		}else if(cluster instanceof XMeans){
			modelName = "Xmeans";		
		}else if(cluster instanceof EM){
			modelName = "EM";			
		}else if(cluster instanceof Cobweb){
			modelName = "Cobweb";					
		}
		return modelName;
	}


	 /**************************
	  * swing 2d plot ���
	  **************************/
	 public void dataVeiwInstances(int size) throws Exception {
		 ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		 PlotData2D plot = new PlotData2D(this.data);
		 vmc.addPlot(plot)	;	 
		 String graphName = "";
		 graphName += " ������ = " + size ;
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(vmc);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     System.out.println(plot.getYindex());
	     frame.setVisible(true);
	     System.out.println("See the " + graphName + " plot");
	 }     

	/**************************
	* swing ���̺�� ������Ģ ���
	**************************/
	public void makeTable(SimpleKMeans cluster) throws Exception{
		// �˰��� �� 1�� + �Ҽ� ������ȣ 1�� + �������� 1�� + �������� 1�� + �Ӽ�����
		String[] header = new String[data.firstInstance().numAttributes()+3];
		header[0] = this.getClusterModelName(cluster); // �˰��� ��
		header[1] = "�Ҽӱ�����ȣ";
		header[2] = "��������";
		for(int x=0; x < data.firstInstance().numAttributes() ; x++){
			header[3+x] = data.firstInstance().attribute(x).name();	// �Ӽ���
		}
		
		// header ���� x �����ͰǼ�
		String[][] contents = new String[data.numInstances()]
				                        [data.firstInstance().numAttributes()+3];


		for (int i = 0; i < data.numInstances(); i++) {
			// �˰��� ��
			contents[i][0] = this.getClusterModelName(cluster); 
			// �Ҽӱ�����ȣ
			contents[i][1] = "cluster :" 
			               + this.data.instance(i).value(
			                 this.data.instance(i).numAttributes()-1); 
			// ��������
			contents[i][2] = "" +(i+1); 
			for (int y=0 ; y < this.data.instance(i).numAttributes(); y++){
				contents[i][3+y] = ""+this.data.instance(i).value(y); // �Ӽ���
			}
		}	
		
		Dimension dim = new Dimension(1500,1000);
		JFrame frame = new JFrame(this.getClusterModelName(cluster) + 
				                " : ������ = " + cluster.getNumClusters());
		frame.setLocation(10, 10);
		frame.setPreferredSize(dim);
			 
		JTable table = new JTable(contents, header);
		JScrollPane scrollpane = new JScrollPane(table);
		frame.add(scrollpane);
		frame.pack();
		frame.setVisible(true);
	}
}
