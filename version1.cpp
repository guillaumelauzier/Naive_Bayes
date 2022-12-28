#include <iostream>
#include <unordered_map>
#include <vector>

// Define a data point with three features (x1, x2, and x3) and a label
struct DataPoint {
    int x1;
    int x2;
    int x3;
    int label;
};

int main() {
    // Create a vector of training data points
    std::vector<DataPoint> trainingData = {{1, 0, 1, 0}, {1, 1, 1, 0}, {1, 0, 0, 1}, {1, 1, 0, 1}};

    // Calculate the prior probabilities for each label
    std::unordered_map<int, float> priors;
    for (const auto& point : trainingData) {
        priors[point.label]++;
    }
    for (const auto& [label, count] : priors) {
        priors[label] = count / trainingData.size();
    }

    // Calculate the conditional probabilities for each feature
    std::unordered_map<int, std::unordered_map<int, float>> conditionals;
    for (const auto& point : trainingData) {
        conditionals[point.label][point.x1]++;
        conditionals[point.label][point.x2]++;
        conditionals[point.label][point.x3]++;
    }
    for (const auto& [label, featureCounts] : conditionals) {
        for (const auto& [feature, count] : featureCounts) {
            conditionals[label][feature] = count / trainingData.size();
        }
    }

    // Define the input data point
    DataPoint input = {1, 1, 0};

    // Calculate the likelihoods for each label
    std::unordered_map<int, float> likelihoods;
    for (const auto& [label, prior] : priors) {
        likelihoods[label] = prior;
        likelihoods[label] *= conditionals[label][input.x1];
        likelihoods[label] *= conditionals[label][input.x2];
        likelihoods[label] *= conditionals[label][input.x3];
    }

    // Predict the label for the input point as the label with the highest likelihood
    int predictedLabel = 0;
    float maxLikelihood = 0;
    for (const auto& [label, likelihood] : likelihoods) {
        if (likelihood > maxLikelihood) {
            maxLikelihood = likelihood;
            predictedLabel = label;
        }
    }

    std::cout << "Predicted label for input point: " << predictedLabel << std::endl;

    return 0;
}
