#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include "Tensor.hpp"

template <typename T>
class Node {
public:
    virtual std::shared_ptr<Tensor<T>> forward() = 0;
    virtual ~Node() = default;
};

#endif // NODE_HPP