import math


def sigmoid(x: float, gain: float = 2.5) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-gain * x))


def grn_step(
    model,
    state,
    dt=0.08,
    gain=1.2,
    threshold=0.15,
    decay=0.03,
):
    """
    SQUAD-inspired continuous GRN update.

    Logic:
    1. Sensor/input nodes are kept fixed during the update.
    2. Positive edge weights act as activation.
    3. Negative edge weights act as inhibition.
    4. The net regulatory input is passed through a sigmoid.
    5. Node states relax gradually toward the target value.

    This is not an exact reproduction of the original SQUAD paper,
    but it is biologically more meaningful than a direct linear update.
    """

    new_state = state.copy()

    for node_idx in range(len(state)):
        if node_idx in model.input_indices:
            continue

        activation_sum = 0.0
        inhibition_sum = 0.0
        incoming_count = 0

        for edge_idx in range(len(model.edges_src)):
            src = model.edges_src[edge_idx]
            dst = model.edges_dst[edge_idx]
            weight = model.edges_weight[edge_idx]

            if dst != node_idx:
                continue

            incoming_count += 1
            signal = weight * state[src]

            if weight >= 0:
                activation_sum += signal
            else:
                inhibition_sum += abs(signal)

        if incoming_count == 0:
            target = 0.0
        else:
            net_input = activation_sum - inhibition_sum
            target = sigmoid(net_input - threshold, gain=gain)

        relaxed = state[node_idx] + dt * (target - state[node_idx])
        decayed = relaxed * (1.0 - decay * dt)

        new_state[node_idx] = max(0.0, min(1.0, decayed))

    return new_state


def run_grn(model, initial_state, steps=10, dt=0.08):
    state = initial_state.copy()
    history = [state.copy()]

    for _ in range(steps):
        state = grn_step(model, state, dt=dt)
        history.append(state.copy())

    return history