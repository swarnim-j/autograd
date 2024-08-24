#ifndef INPUT_NODE_HPP
#define INPUT_NODE_HPP

#include "Node.hpp"

template <typename T>
class InputNode : public Node<T> {
public:
    InputNode(const std::shared_ptr<Tensor<T>>& input);
    std::shared_ptr<Tensor<T>> forward();

private:
    std::shared_ptr<Tensor<T>> input;
};

#endif // INPUT_NODE_HPP

template <typename T>
InputNode<T>::InputNode(const std::shared_ptr<Tensor<T>>& input) : input(input) {}

template <typename T>
std::shared_ptr<Tensor<T>> InputNode<T>::forward() {
    return input;
}