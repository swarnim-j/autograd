#ifndef OPERATION_NODE_HPP
#define OPERATION_NODE_HPP

#include "Node.hpp"
#include <vector>
#include <functional>

template <typename T>
class OperationNode : public Node<T> {
public:
    OperationNode(const std::vector<std::shared_ptr<Node<T>>>& inputs,
    const std::function<std::shared_ptr<Tensor<T>>(const std::vector<std::shared_ptr<Tensor<T>>>&)>& operation);
    std::shared_ptr<Tensor<T>> forward() override;

private:
    std::vector<std::shared_ptr<Node<T>>> inputs;
    std::function<std::shared_ptr<Tensor<T>>(const std::vector<std::shared_ptr<Tensor<T>>>&)> operation;
};

#endif // OPERATION_NODE_HPP