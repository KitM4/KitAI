using KitAI.Regressions.Interfaces;

namespace KitAI.Regressions.ActivationFunctions;

/// <summary>
/// Represents the Sigmoid activation function.
/// </summary>
public class Sigmoid : IActivationFunction
{
    /// <summary>
    /// Activates the Sigmoid function for the given input.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The result of the Sigmoid activation function.</returns>
    public double Activate(double x) =>
        1d / (1d + Math.Exp(-x));
}