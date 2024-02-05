using KitAI.Common.Interfaces;

namespace KitAI.Common.ActivationFunctions;

/// <summary>
/// Represents a step activation function for a perceptron model.
/// </summary>
public class Step : IActivationFunction
{
    /// <summary>
    /// Activates the step function for the given input.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>1 if the input is greater than 0; otherwise, -1.</returns>
    public double Activate(double x) =>
        x > 0 ? 1 : -1;
}