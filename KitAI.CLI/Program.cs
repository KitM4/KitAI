using KitAI.Regressions.Models;
using KitAI.Perceptrons.Models;
using KitAI.Common.Interfaces.Models;
using KitAI.Common.ActivationFunctions;

// ============================================================================================================================================================

Console.WriteLine("=== KitAI Linear regression ===");
Console.WriteLine();
Console.WriteLine(
    "Explanation: Every month we added money to our bank deposit and reinvested the earnings." +
    "\nTask: Based on the current information, predict the amount of income if the amount of our deposit is $21,414.13.");
Console.WriteLine();

double[] depositSet =
[
    7800.0,
    10470.54,
    13158.94,
    15884.40,
    18637.80,
];

double[] profitSet =
[
    70.54,
    88.4,
    125.46,
    153.40,
    176.33,
];

double depositSize = 21414.13;

IRegressionModel linearRegression = new LinearRegression();
linearRegression.Train(depositSet, profitSet);

Console.WriteLine($"* The projected profit will be: ${linearRegression.Predict(depositSize):F2}");
Console.WriteLine();

// ============================================================================================================================================================

Console.WriteLine("=== KitAI Logistic regression ===");
Console.WriteLine();
Console.WriteLine(
    "Explanation: we have students who spent some time preparing for the exams, and after writing the exam" +
    "\nwe have information about who passed and who failed.\nTask: based on the information received\n" +
    "predict whether a student will pass the exam if he or she spends 2.5 hours preparing");
Console.WriteLine();

double[] hours = [0, 1, 2, 3, 4, 5, 6];
double[] passed = [0, 0, 0, 1, 1, 1, 1];

IRegressionModel logisticRegression = new LogisticRegression(new Sigmoid());
logisticRegression.Train(hours, passed);

double testHours = 2.5d;
double prediction = logisticRegression.Predict(testHours);

Console.WriteLine($"* If you spend {testHours} hours studying, you will pass the exam with a {prediction * 100:F2}% chance");
Console.WriteLine();

// ============================================================================================================================================================

Console.WriteLine("=== KitAI Single layer perceptron ===");
Console.WriteLine();
Console.WriteLine("The example demonstrates training the perceptron to learn the logical AND function");
Console.WriteLine();

int inputSize = 2;
double learningRate = 0.01;

IPerceptronModel perceptron = new SingleLayerPerceptron(new Step(), inputSize, learningRate);
perceptron.RandomizeWeights();

double[][] trainingInputs =
[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

int[] targets = [-1, -1, -1, 1];

for (int epoch = 0; epoch < 1000; epoch++)
{
    for (int i = 0; i < trainingInputs.Length; i++)
        perceptron.Train(trainingInputs[i], targets[i]);
}

for (int i = 0; i < trainingInputs.Length; i++)
{
    double output = perceptron.Compute(trainingInputs[i]);
    Console.WriteLine($"Input: [{string.Join(", ", trainingInputs[i])}], Prediction: {output}");
}
Console.WriteLine();

// ============================================================================================================================================================

Console.WriteLine("Press any key to close the program...");
Console.ReadKey(false);