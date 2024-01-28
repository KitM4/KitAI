namespace KitAI.Regressions.Interfaces;

/// <summary>
/// Represents an interface for regression models.
/// </summary>
public interface IRegressionModel
{
    /// <summary>
    /// Trains the regression model using the provided input and output data.
    /// </summary>
    /// <param name="inputData">The input data used for training.</param>
    /// <param name="outputData">The corresponding output data used for training.</param>
    public void Train(double[] inputData, double[] outputData);

    /// <summary>
    /// Predicts the output for the given input data using the trained regression model.
    /// </summary>
    /// <param name="inputData">The input data for prediction.</param>
    /// <returns>The predicted output value.</returns>
    public double Predict(double inputData);
}