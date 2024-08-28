#ifndef OPERATION_H
#define OPERATION_H

#include <memory>
#include <vector>

template<typename T>
class Tensor;

template<typename T>
class Operation : public std::enable_shared_from_this<Operation<T>> {
public:
    std::vector<std::shared_ptr<Tensor<T>>> saved_tensors;

    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) = 0;
    virtual ~Operation() = default;

protected:
    void save_tensors(const std::vector<std::shared_ptr<Tensor<T>>>& tensors);
};

template<typename T>
void Operation<T>::save_tensors(const std::vector<std::shared_ptr<Tensor<T>>>& tensors) {
    saved_tensors = tensors;
}

#endif // OPERATION_H