package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;


import weka.core.Instances;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class MainHW5 {

	private static final int numberOfFolds = 5;
	private static final double alpha = 1.5;

	enum eKernelType {
		RBFKernel,
		PolynomialKernel,
		None
	}

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances randomData = loadData("/Users/elron/Desktop/Machine Learning/HW5/cancer.txt");
		randomData.randomize(new Random());
		Instances trainingDataSet =randomData.trainCV(numberOfFolds, 0);
		Instances testDataSet = randomData.testCV(numberOfFolds, 0);

		SVM smo = new SVM();
		PolyKernel polynomialKernel = new PolyKernel();
		RBFKernel RBFKernel = new RBFKernel();

		int[] polyKernelDegrees = {2, 3, 4};
		double[] gammaValuesArray = {1.0 / 200.0, 1.0 / 20.0, 1.0 / 2.0};

		double bestKernelMeasureResult = Double.MIN_VALUE;
		double bestParameter = 0;		
		eKernelType bestKernelType = eKernelType.None;

		int[] confusionArray;
		double kernelMeasureResult;
		double TPR;
		double FPR;

		for (int currentDegree : polyKernelDegrees){
			polynomialKernel.setExponent(currentDegree);
			smo.setKernel(polynomialKernel);
			smo.buildClassifier(trainingDataSet);

			confusionArray = smo.calcConfusion(testDataSet);
			TPR = calcRecall(confusionArray[0], confusionArray[3]);
			FPR = calcRecall(confusionArray[1], confusionArray[2]);
			kernelMeasureResult = alpha * TPR - FPR;

			if(kernelMeasureResult > bestKernelMeasureResult)
			{
				bestKernelMeasureResult = kernelMeasureResult;
				bestParameter = currentDegree;
				bestKernelType = eKernelType.PolynomialKernel;
			}

			System.out.printf("For PolyKernel with degree %d the rates are:\nTPR = %1.3f\nFPR = %1.3f\n\n",
					currentDegree, TPR, FPR);
		}

		for (double currentGamma : gammaValuesArray){
			RBFKernel.setGamma(currentGamma);
			smo.setKernel(RBFKernel);
			smo.buildClassifier(trainingDataSet);

			confusionArray = smo.calcConfusion(testDataSet);
			TPR = calcRecall(confusionArray[0], confusionArray[3]);
			FPR = calcRecall(confusionArray[1], confusionArray[2]);
			kernelMeasureResult = alpha * TPR - FPR;

			if (kernelMeasureResult > bestKernelMeasureResult){
				bestParameter = currentGamma;
				bestKernelType = eKernelType.RBFKernel;
			}

			System.out.printf("For RBFKernel with gamma %1.3f the rates are:\nTPR = %1.3f\nFPR = %1.3f\n\n"
					,currentGamma, TPR, FPR);
		}

		System.out.printf("The best kernel is: %s %1.3f\n%1.3f\n\n",
				bestKernelType.toString(), bestParameter, bestKernelMeasureResult);

		double currentCValue = 0;
		kernelMeasureResult = 0;
		bestKernelMeasureResult = 0;
		double[] iValues = {1, 0, -1, -2, -3, -4};
		double[] jValues = {3, 2, 1};

		switch (bestKernelType) {
		case PolynomialKernel:
			polynomialKernel.setExponent(bestParameter);
			smo.setKernel(polynomialKernel);
			break;

		case RBFKernel:
			RBFKernel.setGamma(bestParameter);
			smo.setKernel(RBFKernel);
			break;

		default:
			break;
		}

		for (double i : iValues){
			for (double j : jValues){
				currentCValue = (double)Math.pow(10.0, i) * (double)(j / 3);
				smo.setC(currentCValue);
				smo.buildClassifier(trainingDataSet);
				confusionArray = smo.calcConfusion(testDataSet);
				TPR = calcRecall(confusionArray[0], confusionArray[3]);
				FPR = calcRecall(confusionArray[1], confusionArray[2]);

				if(kernelMeasureResult > bestKernelMeasureResult)
				{
					bestKernelMeasureResult = kernelMeasureResult;
				}
				
				System.out.println("For C " + currentCValue + " the rates are:\n" + "TPR = " + TPR + "\nFPR = " + FPR + "\n");
			}
		}
	}

	// calculation of TPR/FPR
	private static double calcRecall(int i_Num1, int i_Num2)
	{
		double result = 0;
		
		if(i_Num1 + i_Num2 != 0)
		{
			result = (double)i_Num1 / (double)(i_Num1 + i_Num2);
		}
		
		return result;
	}
}
