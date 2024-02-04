using KitAI.Common.Interfaces.Models;

namespace KitAI.Regressions.Models;

/// <summary>
/// Represents a linear regression model.
/// </summary>
public class LinearRegression : IRegressionModel
{
    /// <summary>
    /// Gets the coefficient B1 of the linear regression model.
    /// </summary>
    public double B1 { get; private set; } = 0d;

    /// <summary>
    /// Gets the intercept B0 of the linear regression model.
    /// </summary>
    public double B0 { get; private set; } = 0d;

    /// <summary>
    /// Trains the linear regression model using the provided input and output data.
    /// </summary>
    /// <param name="inputData">The input data used for training.</param>
    /// <param name="outputData">The corresponding output data used for training.</param>
    public void Train(double[] inputData, double[] outputData)
    {
        double xAverage = GetAverageValue(inputData);
        double yAverage = GetAverageValue(outputData);

        CalculationB1Coefficient(inputData, outputData, xAverage, yAverage);
        CalculationB0Coefficient(xAverage, yAverage);
    }

    /// <summary>
    /// Predicts the output for the given input data using the trained linear regression model.
    /// </summary>
    /// <param name="inputData">The input data for prediction.</param>
    /// <returns>The predicted output value.</returns>
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