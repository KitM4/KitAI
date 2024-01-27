using KitAI.Regressions.Interfaces;

namespace KitAI.Regressions.ActivationFunctions;

public class Sigmoid : IActivationFunction
{
    public double Activate(double x) =>
        1d / (1d + Math.Exp(-x));
}