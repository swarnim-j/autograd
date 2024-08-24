#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    // Constructor
    Tensor(const std::vector<T>& data, const std::vector<size_t>& dims);

    // Operations on tensor
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& other);

    void print() const;

private:
    std::vector<T> data;  // Tensor data
    std::vector<size_t> dims;  // Tensor dimensions

    // Helper functions
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    std::vector<size_t> get_strides() const;
};

// Constructor: initialize the tensor with data and dimensions
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& dimensions) : data(data), dims(dimensions) {
    size_t expected_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    if (data.size() != expected_size) {
        throw std::invalid_argument("Data size does not match tensor dimensions.");
    }
}

// Get flat index from multi-dimensional indices
template <typename T>
size_t Tensor<T>::get_flat_index(const std::vector<size_t>& indices) const {
    const auto strides = get_strides();
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}

// Get strides for multi-dimensional indexing
template <typename T>
std::vector<size_t> Tensor<T>::get_strides() const {
    std::vector<size_t> strides(dims.size());
    size_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }
    return strides;
}

// Element-wise addition
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator+(const std::shared_ptr<Tensor<T>>& other) {
    if (dims != other->dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }

    std::vector<T> result(data.size());
    std::transform(data.begin(), data.end(), other->data.begin(), result.begin(), std::plus<T>());

    return std::make_shared<Tensor<T>>(result, dims);
}

// Element-wise subtraction
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator-(const std::shared_ptr<Tensor<T>>& other) {
    if (dims != other->dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
    }

    std::vector<T> result(data.size());
    std::transform(data.begin(), data.end(), other->data.begin(), result.begin(), std::minus<T>());

    return std::make_shared<Tensor<T>>(result, dims);
}

// Element-wise multiplication
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator*(const std::shared_ptr<Tensor<T>>& other) {
    if (dims != other->dims) {
        throw std::invalid_argument("Tensors must have the same dimensions for multiplication.");
    }

    std::vector<T> result(data.size());
    std::transform(data.begin(), data.end(), other->data.begin(), result.begin(), std::multiplies<T>());

    return std::make_shared<Tensor<T>>(result, dims);
}

// Print tensor values
template <typename T>
void Tensor<T>::print() const {
    std::cout << "Tensor dimensions: [ ";
    for (const auto& dim : dims) {
        std::cout << dim << " ";
    }
    std::cout << "]\n";

    size_t offset = 0;
    for (const auto& val : data) {
        std::cout << val << " ";
        if (++offset % dims.back() == 0) {
            std::cout << "\n";
        }
    }
}

#endif // TENSOR_H
