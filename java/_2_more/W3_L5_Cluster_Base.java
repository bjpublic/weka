package _2_more;

import java.awt.*;
import java.io.*;
import javax.swing.*;
import weka.clusterers.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.*;

public class W3_L5_Cluster_Base {
	Instances data=null;
	ClusterEvaluation eval=null;
	
	public W3_L5_Cluster_Base (String fileName) throws Exception{
		// 1) data loader
		String folderName = "D:\\Weka-3-9\\data\\";
		this.data=new Instances(
				  new BufferedReader(
				  new FileReader(folderName+fileName+".arff")));
	}

	public static void main(String args[]) throws Exception{
		new W3_L5_Cluster_Base("iris").cluster(new SimpleKMeans(), 3);
		new W3_L5_Cluster_Base("iris").cluster(new XMeans(), 3);
		new W3_L5_Cluster_Base("iris").cluster(new EM(), 3);
		new W3_L5_Cluster_Base("iris").cluster(new Cobweb(), 0.6);
	}

	public void cluster(Clusterer cluster, double k) throws Exception{
		
		// 2) class assigner ���ʿ� , ��� class �� ����
		Remove filter = new Remove();
		filter.setAttributeIndices("last");
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);		
		
		// 3) Evaluation ��ü ����
		this.eval=new ClusterEvaluation();
		
		// 4) model run
		// 5) evaluate
		// 6) print result
		System.out.println("***************************************");
		this.goCluster(cluster, k );
		
		// 7) ������ ������ ���� ��� (���� ����/�ٿ��ֱ� ����)
//		System.out.println(cluster);
		System.out.println(eval.clusterResultsToString());
		this.makeTable(cluster);
		System.out.println("***************************************");
		

	}

	/**************************
	* �����м� ��ü���� ó������� �����Ͽ� ĳ������ ���� �Լ��б�
	**************************/
	public void goCluster(Clusterer cluster, double k) throws Exception{
		if(cluster instanceof SimpleKMeans){
			this.Kmeasns((SimpleKMeans)cluster, (int) k);
		}else if(cluster instanceof XMeans){
			this.Xmeans((XMeans)cluster, (int) k);			
		}else if(cluster instanceof EM){
			this.EM((EM)cluster, (int) k);			
		}else if(cluster instanceof Cobweb){
			this.cobweb((Cobweb)cluster , k);			
		}
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
	
	public void Kmeasns(SimpleKMeans kmeans, int k) throws Exception{

		// 4) model run
		kmeans.setNumClusters(k);
		kmeans.buildClusterer(this.data);
		
		// 5) evaluate
		eval.setClusterer(kmeans);
		eval.evaluateClusterer(this.data);
		
		// 6) print Result text
		System.out.println("���� �˰��� :" + 
		                   this.getClusterModelName(kmeans) );
		System.out.println("������ :" + kmeans.numberOfClusters() );
		System.out.println("kmeans ����ǥ-�ּҿ��������� : " + 
		                    kmeans.getSquaredError());
		double d[] = kmeans.getClusterSizes();
	}
	
	/**
	 * KD-Tree �� ���� �����̳� ǰ����ǥ�� ���� ���� ������ �� �˰����̴�.
	 * */
	public void Xmeans(XMeans xmeans, int k) throws Exception{
		// 4) model run
		xmeans.setMinNumClusters(k);
		xmeans.setMaxNumClusters(k);
		xmeans.buildClusterer(this.data);
		
		// 5) evaluate
		eval.setClusterer(xmeans);
		eval.evaluateClusterer(data);
		
		// 6) print Result text
		System.out.println("���� �˰��� :" + 
                            this.getClusterModelName(xmeans) );
		System.out.println("������ :" + xmeans.numberOfClusters() );
	}
	
	public void EM(EM em, int k) throws Exception{
		// 4) model run
		em.setNumClusters(k);
		em.buildClusterer(this.data);
		
		// 5) evaluate
		eval.setClusterer(em);
		eval.evaluateClusterer(data);
		
		// 6) print Result text
		System.out.println("���� �˰��� :" + 
		                   this.getClusterModelName(em) );
		System.out.println("������ :" + em.numberOfClusters() );
		System.out.println("em ����ǥ-�α׿쵵 : " + 
                           this.eval.getLogLikelihood());
	}
	
	public void cobweb (Cobweb cobweb, double k) throws Exception{
		// 4) model run
		cobweb.setAcuity(k);		
		cobweb.buildClusterer(this.data);
		
		// 5) evaluate
		eval.setClusterer(cobweb);
		eval.evaluateClusterer(data);
		
		// 6) print Result text
		System.out.println("���� �˰��� :" +  
		                    this.getClusterModelName(cobweb) );
		System.out.println("������ :" + cobweb.numberOfClusters() );
		
		this.treeVeiwInstances(cobweb);
	}
	

	 /**************************
	  * cobweb ����α׷� ���
	  **************************/
	 public void treeVeiwInstances(Cobweb cobweb) throws Exception {

		 String graphName = "";
		 graphName += " cobweb ����α׷� ";
	     TreeVisualizer panel = new TreeVisualizer(null,
	    		                    cobweb.graph(),
	    		                    new PlaceNode2());
	     JFrame frame = new JFrame(graphName);
	     frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	     frame.getContentPane().setLayout(new BorderLayout());
	     frame.getContentPane().add(panel);
	     frame.setSize(new Dimension(800,500));
	     frame.setLocationRelativeTo(null);
	     frame.setVisible(true);
	     panel.fitToScreen();
	     System.out.println("See the " + graphName + " plot");
	 }     

	/**************************
	* swing ���̺�� ������Ģ ���
	**************************/
	public void makeTable(Clusterer cluster) throws Exception{
		// �˰��� �� 1�� + �Ҽ� ������ȣ 1�� + �������� 1�� + �������� 1�� + �Ӽ�����
		String[] header = new String[data.firstInstance().numAttributes()+4];
		header[0] = this.getClusterModelName(cluster); // �˰��� ��
		header[1] = "�Ҽӱ�����ȣ";
		header[2] = "��������";
		header[3] = "��������";
		for(int x=0; x < data.firstInstance().numAttributes() ; x++){
			header[4+x] = data.firstInstance().attribute(x).name();	// �Ӽ���
		}
		
		// header ���� x �����ͰǼ�
		String[][] contents = new String[data.numInstances()]
				                        [data.firstInstance().numAttributes()+4];

		int row=0;
		for (int kk=0 ; kk < eval.getNumClusters() ; kk++){
			int clusterRow = 0;
			for (int i = 0; i < data.numInstances(); i++) {
				if(kk == cluster.clusterInstance(this.data.instance(i)) ){
					contents[row][0] = this.getClusterModelName(cluster); // �˰��� ��
					contents[row][1] = "cluster :" + kk; // �Ҽӱ�����ȣ
					contents[row][2] = "" +(++clusterRow); // ��������
					contents[row][3] = "" +(i+1); // ��������
					for (int y=0 ; y < this.data.instance(i).numAttributes(); y++){
						contents[row][4+y] = ""+this.data.instance(i).value(y); // �Ӽ���
					}
					row++;
				}
			}	
		}
		
		Dimension dim = new Dimension(1500,1000);
		JFrame frame = new JFrame(this.getClusterModelName(cluster) + 
				                " : ������ = " + eval.getNumClusters());
		frame.setLocation(10, 10);
		frame.setPreferredSize(dim);
			 
		JTable table = new JTable(contents, header);
		JScrollPane scrollpane = new JScrollPane(table);
		frame.add(scrollpane);
		frame.pack();
		frame.setVisible(true);
	}
}
