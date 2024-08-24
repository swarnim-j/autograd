#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "Node.hpp"
#include "InputNode.hpp"
#include "OperationNode.hpp"

template <typename T>
class Graph {
public:
    std::shared_ptr<Node<T>> add_input(std::shared_ptr<Tensor<T>> input);
    std::shared_ptr<Node<T>> add_operation(
        const std::vector<std::shared_ptr<Node<T>>>& inputs,
        const std::function<std::shared_ptr<Tensor<T>>(const std::vector<std::shared_ptr<Tensor<T>>>&)>& operation
    );
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Node<T>> output_node);

private:
    std::vector<std::shared_ptr<Node<T>>> nodes;
};

#endif // GRAPH_HPP

template <typename T>
std::shared_ptr<Node<T>> Graph<T>::add_input(std::shared_ptr<Tensor<T>> input) {
    auto node = std::make_shared<InputNode<T>>(input);
    nodes.push_back(node);
    return node;
}

template <typename T>
std::shared_ptr<Node<T>> Graph<T>::add_operation(
    const std::vector<std::shared_ptr<Node<T>>>& inputs,
    const std::function<std::shared_ptr<Tensor<T>>(const std::vector<std::shared_ptr<Tensor<T>>>&)>& operation
) {
    auto node = std::make_shared<OperationNode<T>>(inputs, operation);
    nodes.push_back(node);
    return node;
}

template <typename T>
std::shared_ptr<Tensor<T>> Graph<T>::forward(std::shared_ptr<Node<T>> output_node) {
    return output_node->forward();
}