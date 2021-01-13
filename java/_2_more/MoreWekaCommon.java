package _2_more;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.filters.unsupervised.instance.imagefilter.*;

public class MoreWekaCommon {
	public final static String modelPath ="D:\\Weka-3-9\\model\\";
	public final static String arffPath ="D:\\Weka-3-9\\data\\";
	public final static String imgPath ="D:\\Weka-3-9\\data\\image";
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	/*****************************
	 * Model Name
	 *****************************/
	public static String getModelName(Classifier model){
		String modelName = "";
		if ( model instanceof  Logistic)
			modelName = "Logistic";
		else if ( model instanceof  J48)
			modelName = "J48";
		else if ( model instanceof  J48)
			modelName = "J48";
		else if ( model instanceof  IBk)
			modelName = "IBk";
		else if ( model instanceof  SMO)
			modelName = "SMO";
		else if ( model instanceof  OneR)
			modelName = "OneR";
		else if ( model instanceof  NaiveBayes)
			modelName = "NaiveBayes";
		else if ( model instanceof  DecisionStump)
			modelName = "DecisionStump";
		else if ( model instanceof  ZeroR)
			modelName = "ZeroR";
		else if ( model instanceof  FilteredClassifier)
			modelName = "FilteredClassifier";
		else if ( model instanceof  NaiveBayesMultinomial)
			modelName = "NaiveBayesMultinomial";
			
		return modelName;
	}
	

	/*****************************
	 * AttributeSelection Name
	 *****************************/
	public static String getFeatureAlgorithmName(ASEvaluation attrSel){
		String modelName = "";
		if ( attrSel instanceof  WrapperSubsetEval)
			modelName = "WrapperSubsetEval";
		else if ( attrSel instanceof  CorrelationAttributeEval)
			modelName = "CorrelationAttributeEval";
		else if ( attrSel instanceof  GainRatioAttributeEval)
			modelName = "GainRatioAttributeEval";
		else if ( attrSel instanceof  InfoGainAttributeEval)
			modelName = "InfoGainAttributeEval";
		else if ( attrSel instanceof  OneRAttributeEval)
			modelName = "OneRAttributeEval";
		return modelName;
	}
	
	public static String getImageFilterName (AbstractImageFilter imageFilter){
		String filterName = "";
		if ( imageFilter instanceof  ColorLayoutFilter)
			filterName = "ColorLayoutFilter";
		else if ( imageFilter instanceof  EdgeHistogramFilter)
			filterName = "EdgeHistogramFilter";
		else if ( imageFilter instanceof  SimpleColorHistogramFilter)
			filterName = "SimpleColorHistogramFilter";
		
		return filterName;
	}
}
