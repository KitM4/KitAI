namespace KitAI.Regressions.Interfaces;

public interface IRegressionModel
{
    public void Train(double[] inputData, double[] outputData);

    public double Predict(double inputData);
}