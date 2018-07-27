package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}

	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}

	public int[] calcConfusion(Instances instances) throws Exception{
		double classifyInstanceValue;
		int TP = 0, FP = 0, TN = 0, FN = 0;
		for (Instance currentInstance : instances){
			classifyInstanceValue = m_smo.classifyInstance(currentInstance);
			
			if (currentInstance.classValue() == 1){
				if (classifyInstanceValue == 1){
					TP++;
				}
				else{
					FN++;
				}	
			}
			else{
				if (classifyInstanceValue == 1){
					FP++;
				}
				else{
					TN++;
				}
			}
		}

		int [] calcConfusionOutPut = {TP, FP, TN, FN};
		return calcConfusionOutPut;
	}

	public void setKernel(Kernel i_KernelToSet) {
		m_smo.setKernel(i_KernelToSet);
	}

	public void setC(double i_CValueToSet) {
		m_smo.setC(i_CValueToSet);
	}

	public double getC() {
		return m_smo.getC();
	}
}
