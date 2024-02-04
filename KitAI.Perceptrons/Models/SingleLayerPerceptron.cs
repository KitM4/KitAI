using KitAI.Common.Interfaces;
using KitAI.Common.Interfaces.Models;

namespace KitAI.Perceptrons.Models;

public class SingleLayerPerceptron(IActivationFunction activationFunction, int inputCount, double learningRate) : IPerceptronModel
{
    private readonly IActivationFunction _activationFunction = activationFunction;

    private readonly double[] _weights = new double[inputCount];
    private readonly double _learningRate = learningRate;
    private double _bias = 0d;

    public void InitializeWeights()
    {
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = 1d;
    }

    public void RandomizeWeights()
    {
        Random random = new((int)DateTime.UtcNow.Ticks);
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = random.NextDouble();

        _bias = random.NextDouble();
    }

    public double Compute(double[] inputs)
    {
        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * _weights[i];

        sum += _bias;

        return _activationFunction.Activate(sum);
    }

    public void Train(double[] inputs, double target)
    {
        double prediction = Compute(inputs);
        double error = target - prediction;

        for (int i = 0; i < _weights.Length; i++)
            _weights[i] += _learningRate * error * inputs[i];

        _bias += _learningRate * error;
    }
}