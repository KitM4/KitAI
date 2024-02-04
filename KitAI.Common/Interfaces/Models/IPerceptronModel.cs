namespace KitAI.Common.Interfaces.Models;

public interface IPerceptronModel
{
    public void InitializeWeights();

    public void RandomizeWeights();

    public double Compute(double[] inputs);

    public void Train(double[] inputs, double target);
}