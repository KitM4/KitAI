using KitAI.Common.Interfaces;
using KitAI.Common.Interfaces.Models;

namespace KitAI.Perceptrons.Models;

/// <summary>
/// Initializes a new instance of the SingleLayerPerceptron class.
/// </summary>
/// <remarks>
/// This class performs computations for a single-layer perceptron with a specified activation function,
/// input count, and learning rate.
/// </remarks>
public class SingleLayerPerceptron(IActivationFunction activationFunction, int inputCount, double learningRate) : IPerceptronModel
{
    private readonly IActivationFunction _activationFunction = activationFunction;

    private readonly double[] _weights = new double[inputCount];
    private readonly double _learningRate = learningRate;
    private double _bias = 0d;

    /// <summary>
    /// Initializes the weights with default values.
    /// </summary>
    public void InitializeWeights()
    {
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = 1d;
    }

    /// <summary>
    /// Randomizes the weights and bias of the perceptron.
    /// </summary>
    public void RandomizeWeights()
    {
        Random random = new((int)DateTime.UtcNow.Ticks);
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = random.NextDouble();

        _bias = random.NextDouble();
    }

    /// <summary>
    /// Computes the output of the perceptron for the given inputs.
    /// </summary>
    /// <param name="inputs">The input values to the perceptron.</param>
    /// <returns>The computed output of the perceptron.</returns>
    public double Compute(double[] inputs)
    {
        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * _weights[i];

        sum += _bias;

        return _activationFunction.Activate(sum);
    }

    /// <summary>
    /// Trains the perceptron using the provided inputs and target output.
    /// </summary>
    /// <param name="inputs">The input values to the perceptron.</param>
    /// <param name="target">The target output for training.</param>
    public void Train(double[] inputs, double target)
    {
        double prediction = Compute(inputs);
        double error = target - prediction;

        for (int i = 0; i < _weights.Length; i++)
            _weights[i] += _learningRate * error * inputs[i];

        _bias += _learningRate * error;
    }
}