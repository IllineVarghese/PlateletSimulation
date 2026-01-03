import numpy as np
import warp as wp

wp.init()
device = "cpu"

@wp.kernel
def add_one(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] += 1.0

def main():
    n = 10
    a = wp.array(np.zeros(n, dtype=np.float32), dtype=wp.float32, device=device)
    wp.launch(add_one, dim=n, inputs=[a], device=device)
    print(a.numpy())

if __name__ == "__main__":
    main()
