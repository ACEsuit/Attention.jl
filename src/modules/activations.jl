module Activations


function native_softmax(x, dims)
    exp_x = exp.(x)
    sum_exp_x = sum(exp_x, dims=dims)
    exp_x ./ sum_exp_x
end

function matrix_dropout(x, rate)
    if rate == 0.0
        return x
    end
    mask = rand(Float32, size(x)) .> rate
    x .* mask / (1 - rate)
end

end