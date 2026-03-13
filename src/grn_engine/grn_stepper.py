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

        target = sigmoid(input_sum - 1.0)
   

        new_state[i] = state[i] + dt * (target - state[i])

    return new_state

def run_grn(model, initial_state, steps=10):
    """
    Run GRN dynamics for multiple time steps.

    model : GRNModel
    initial_state : list of node values
    steps : number of simulation steps
    """

    state = initial_state.copy()
    history = [state]

    for _ in range(steps):
        state = grn_step(model, state)
        history.append(state)

    return history