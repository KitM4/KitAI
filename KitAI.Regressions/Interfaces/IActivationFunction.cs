namespace KitAI.Regressions.Interfaces;

/// <summary>
/// Represents an activation function interface.
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// Activates the function for the given input.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The result of the activation function.</returns>
    public double Activate(double x);
}