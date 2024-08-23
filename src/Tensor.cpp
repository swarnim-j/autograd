#include "Tensor.h"

template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data) : data(data) {}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator+(const std::shared_ptr<Tensor<T>>& other) {
    if (data.size() != other->data.size()) {
        throw std::invalid_argument("Tensors must have the same size for addition.");
    }

    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] + other->data[i];
    }

    return std::make_shared<Tensor<T>>(result);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator-(const std::shared_ptr<Tensor<T>>& other) {
    if (data.size() != other->data.size()) {
        throw std::invalid_argument("Tensors must have the same size for subtraction.");
    }

    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] - other->data[i];
    }

    return std::make_shared<Tensor<T>>(result);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator*(const std::shared_ptr<Tensor<T>>& other) {
    if (data.size() != other->data.size()) {
        throw std::invalid_argument("Tensors must have the same size for multiplication.");
    }

    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * other->data[i];
    }

    return std::make_shared<Tensor<T>>(result);
}

template <typename T>
void Tensor<T>::print() const {
    std::cout << "[ ";
    for (const auto& val : data) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

