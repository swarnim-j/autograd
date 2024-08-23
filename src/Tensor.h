#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    Tensor(const std::vector<T>& data);

    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& other);

    void print() const;

private:
    std::vector<T> data;
};

#endif // TENSOR_H