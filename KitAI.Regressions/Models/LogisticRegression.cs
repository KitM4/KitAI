using KitAI.Regressions.Interfaces;

namespace KitAI.Regressions.Models;

public class LogisticRegression(IActivationFunction activationFunction) : IRegressionModel
{
    public double Weight { get; private set; } = 0d;

    public double Bias { get; private set; } = 0d;

    public double LearningRate { get; set; } = 0.001d;

    public int TrainingIterations { get; set; } = 1000;

    private readonly IActivationFunction _activationFunction = activationFunction;

    public void Train(double[] inputData, double[] outputData)
    {
        for (int i = 0; i < TrainingIterations; i++)
        {
            double totalError = 0.0;

            for (int j = 0; j < inputData.Length; j++)
            {
                double prediction = Predict(inputData[j]);
                double error = outputData[j] - prediction;

                totalError += error * error;

                Weight += LearningRate * error * prediction * (1d - prediction) * inputData[j];
                Bias += LearningRate * error * prediction * (1d - prediction);
            }
        }
    }

    public double Predict(double inputData) =>
        _activationFunction.Activate(Bias + Weight * inputData);
}