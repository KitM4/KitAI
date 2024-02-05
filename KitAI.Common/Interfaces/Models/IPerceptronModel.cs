namespace KitAI.Common.Interfaces.Models;

/// <summary>
/// Represents a perceptron model that can be trained and used for predictions.
/// </summary>
public interface IPerceptronModel
{
    /// <summary>
    /// Initializes the weights of the perceptron.
    /// </summary>
    public void InitializeWeights();

    /// <summary>
    /// Randomizes the weights of the perceptron for better training convergence.
    /// </summary>
    public void RandomizeWeights();

    /// <summary>
    /// Computes the output of the perceptron for the given inputs.
    /// </summary>
    /// <param name="inputs">The input values to the perceptron.</param>
    /// <returns>The computed output of the perceptron.</returns>
    public double Compute(double[] inputs);

    /// <summary>
    /// Trains the perceptron using the provided inputs and target output.
    /// </summary>
    /// <param name="inputs">The input values to the perceptron.</param>
    /// <param name="target">The target output for training.</param>
    public void Train(double[] inputs, double target);
}