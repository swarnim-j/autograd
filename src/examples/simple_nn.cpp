#include <iostream>
#include <vector>
#include <random>
#include "AutogradOps.h"
#include "layers/LinearLayer.h"
#include "optimizers/SGD.h"

// Simple 2-layer neural network for binary classification
template<typename T>
class SimpleNN : public Module<T> {
public:
    SimpleNN(size_t input_size, size_t hidden_size) 
        : layer1(input_size, hidden_size),
          layer2(hidden_size, 1) {
        // Add layers' parameters to the module's parameters
        auto layer1_params = layer1.parameters();
        auto layer2_params = layer2.parameters();
        this->params.insert(this->params.end(), layer1_params.begin(), layer1_params.end());
        this->params.insert(this->params.end(), layer2_params.begin(), layer2_params.end());
    }

    std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input) override {
        auto hidden = AutogradOps<T>::relu(layer1.forward(input));
        auto output = AutogradOps<T>::sigmoid(layer2.forward(hidden));
        return output;
    }

private:
    LinearLayer<T> layer1;
    LinearLayer<T> layer2;
};

// Generate some random training data
template<typename T>
std::pair<std::vector<std::shared_ptr<Tensor<T>>>, std::vector<std::shared_ptr<Tensor<T>>>> 
generate_data(size_t n_samples, size_t input_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1, 1);

    std::vector<std::shared_ptr<Tensor<T>>> inputs;
    std::vector<std::shared_ptr<Tensor<T>>> targets;

    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<T> input_data(input_size);
        for (auto& x : input_data) {
            x = dis(gen);
        }
        
        // Simple rule: if sum of inputs > 0, class is 1, else 0
        T sum = std::accumulate(input_data.begin(), input_data.end(), 0.0);
        T target = sum > 0 ? 1.0 : 0.0;

        inputs.push_back(AutogradOps<T>::tensor(input_data, {input_size}, true));
        targets.push_back(AutogradOps<T>::tensor({target}, {1}, false));
    }

    return {inputs, targets};
}

int main() {
    try {
        // Create a simple neural network
        SimpleNN<float> model(2, 4);  // 2 inputs, 4 hidden units, 1 output
        SGD<float> optimizer(model.parameters(), 0.1);  // learning rate = 0.1

        // Generate training data
        auto [inputs, targets] = generate_data<float>(100, 2);

        // Training loop
        for (int epoch = 0; epoch < 100; ++epoch) {
            float total_loss = 0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward pass
                auto output = model.forward(inputs[i]);
                auto loss = AutogradOps<float>::mse_loss(output, targets[i]);
                total_loss += loss->data[0];

                // Backward pass
                optimizer.zero_grad();
                loss->backward();
                optimizer.step();
            }

            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << (epoch + 1) << ", Average Loss: " 
                         << total_loss / inputs.size() << std::endl;
            }
        }

        // Test the model
        std::cout << "\nTesting the model:\n";
        std::vector<float> test_input = {0.5, -0.5};  // Should be class 0
        auto test_tensor = AutogradOps<float>::tensor(test_input, {2}, true);
        auto prediction = model.forward(test_tensor);
        std::cout << "Input: [0.5, -0.5], Prediction: " << prediction->data[0] << std::endl;

        test_input = {0.8, 0.2};  // Should be class 1
        test_tensor = AutogradOps<float>::tensor(test_input, {2}, true);
        prediction = model.forward(test_tensor);
        std::cout << "Input: [0.8, 0.2], Prediction: " << prediction->data[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 