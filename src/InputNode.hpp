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