#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <memory>
#include "Tensor.h"

template<typename T>
class Module {
public:
    virtual ~Module() = default;

    // Forward pass
    virtual std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input) = 0;

    // Get all parameters of the module
    virtual std::vector<std::shared_ptr<Tensor<T>>> parameters() {
        return params;
    }

    // Zero out all parameter gradients
    virtual void zero_grad() {
        for (auto& param : params) {
            param->zero_grad();
        }
    }

protected:
    std::vector<std::shared_ptr<Tensor<T>>> params;
};

#endif // MODULE_H 