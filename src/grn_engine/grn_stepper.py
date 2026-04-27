import math


def sigmoid(x: float, gain: float = 8.0) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-gain * x))


def grn_step(
    model,
    state,
    dt=0.04,
    gain=3.0,
    threshold=0.60,
    decay=0.03,
):
    """
    SQUAD-inspired continuous GRN update.

    This is not a full validated SQUAD reproduction.
    It is a nonlinear continuous regulatory update:
    weighted input -> sigmoid response -> relaxation dynamics.
    """

    new_state = state.copy()

    for i in range(len(state)):
        if i in model.input_indices:
            continue

        input_sum = 0.0
        incoming_edges = 0

        for e in range(len(model.edges_src)):
            src = model.edges_src[e]
            dst = model.edges_dst[e]

            if dst == i:
                weight = model.edges_weight[e]
                input_sum += weight * state[src]
                incoming_edges += 1

        if incoming_edges == 0:
            target = state[i] * (1.0 - decay)
        else:
            target = sigmoid(input_sum - threshold, gain=gain)

        new_value = state[i] + dt * (target - state[i])
        new_state[i] = max(0.0, min(1.0, new_value))

    return new_state


def run_grn(model, initial_state, steps=10, dt=0.15):
    state = initial_state.copy()
    history = [state.copy()]

    for _ in range(steps):
        state = grn_step(model, state, dt=dt)
        history.append(state.copy())

    return history