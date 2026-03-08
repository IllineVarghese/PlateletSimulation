import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def grn_step(model, state, dt=0.1):
    """
    Perform one GRN update step.

    model : GRNModel
    state : list[float] (node values)
    dt    : relaxation timestep
    """

    new_state = state.copy()

    for i in range(len(state)):

        # skip input nodes (they are controlled by sensors)
        if i in model.input_indices:
            continue

        input_sum = 0.0

        for e in range(len(model.edges_src)):
            src = model.edges_src[e]
            dst = model.edges_dst[e]

            if dst == i:
                weight = model.edges_weight[e]
                input_sum += weight * state[src]

        target = sigmoid(input_sum)

        new_state[i] = state[i] + dt * (target - state[i])

    return new_state