import math


def sigmoid(x: float, gain: float = 2.2) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-gain * x))


def grn_step(
    model,
    state,
    dt=0.025,
    gain=2.2,
    threshold=0.45,
    decay=0.08,
):
    """
    SQUAD-inspired continuous GRN update.

    This is not a fully validated reproduction of the original SQUAD method.
    It implements the core useful idea for this thesis prototype:
    weighted regulation -> nonlinear sigmoid response -> relaxed state update.

    Supports positive and negative edge weights.
    Negative weights act as inhibition.
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
            target = 0.0
        else:
            target = sigmoid(input_sum - threshold, gain=gain)

        relaxed = state[i] + dt * (target - state[i])
        decayed = relaxed * (1.0 - decay * dt)

        new_state[i] = max(0.0, min(1.0, decayed))

    return new_state


def run_grn(model, initial_state, steps=10, dt=0.025):
    state = initial_state.copy()
    history = [state.copy()]

    for _ in range(steps):
        state = grn_step(model, state, dt=dt)
        history.append(state.copy())

    return history