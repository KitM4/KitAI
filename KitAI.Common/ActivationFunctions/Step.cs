using KitAI.Common.Interfaces;

namespace KitAI.Common.ActivationFunctions
{
    public class Step : IActivationFunction
    {
        public double Activate(double x) =>
            x > 0 ? 1 : -1;
    }
}