using KitAI.Regressions.Interfaces;

namespace KitAI.Regressions.Models;

public class LinearRegression : IRegressionModel
{
    public double B1 { get; private set; } = 0d;

    public double B0 { get; private set; } = 0d;

    public void Train(double[] inputData, double[] outputData)
    {
        double xAverage = GetAverageValue(inputData);
        double yAverage = GetAverageValue(outputData);

        CalculationB1Coefficient(inputData, outputData, xAverage, yAverage);
        CalculationB0Coefficient(xAverage, yAverage);
    }

    public double Predict(double inputData) =>
        B0 + B1 * inputData;

    private void CalculationB1Coefficient(double[] x, double[] y, double xAverage, double yAverage)
    {
        double numerator = 0d;
        double denominator = 0d;

        for (int i = 0; i < x.Length; i++)
        {
            numerator += (x[i] - xAverage) * (y[i] - yAverage);
            denominator += Math.Pow(x[i] - xAverage, 2d);
        }

        B1 = numerator / denominator;
    }

    private void CalculationB0Coefficient(double xAverage, double yAverage) =>
        B0 = yAverage - B1 * xAverage;

    private static double GetAverageValue(double[] values) =>
        values.Sum() / values.Length;
}