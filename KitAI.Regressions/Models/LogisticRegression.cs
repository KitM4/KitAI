using KitAI.Common.Interfaces;
using KitAI.Regressions.Interfaces;

namespace KitAI.Regressions.Models;

/// <summary>
/// Represents and initializes a new instance of the logistic regression model.
/// </summary>
/// <param name="activationFunction">The activation function to be used.</param>
public class LogisticRegression(IActivationFunction activationFunction) : IRegressionModel
{
    /// <summary>
    /// Gets the weight parameter of the logistic regression model.
    /// </summary>
    public double Weight { get; private set; } = 0d;

    /// <summary>
    /// Gets the bias parameter of the logistic regression model.
    /// </summary>
    public double Bias { get; private set; } = 0d;

    /// <summary>
    /// Gets or sets the learning rate for training the logistic regression model.
    /// </summary>
    public double LearningRate { get; set; } = 0.001d;

    /// <summary>
    /// Gets or sets the number of training iterations for the logistic regression model.
    /// </summary>
    public int TrainingIterations { get; set; } = 1000;

    private readonly IActivationFunction _activationFunction = activationFunction;

    /// <summary>
    /// Trains the logistic regression model using the provided input and output data.
    /// </summary>
    /// <param name="inputData">The input data used for training.</param>
    /// <param name="outputData">The corresponding output data used for training.</param>
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

    /// <summary>
    /// Predicts the output for the given input data using the trained logistic regression model.
    /// </summary>
    /// <param name="inputData">The input data for prediction.</param>
    /// <returns>The predicted output value.</returns>
    public double Predict(double inputData) =>
        _activationFunction.Activate(Bias + Weight * inputData);
}