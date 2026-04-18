import numpy as np


class ChemicalField:
    def __init__(self, nx=20, ny=20, nz=20, spacing=0.05, decay_rate=0.1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.spacing = spacing
        self.decay_rate = decay_rate
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.values = np.zeros((nx, ny, nz), dtype=np.float32)

    def world_to_grid(self, pos):
        ix = int((float(pos[0]) - self.origin[0]) / self.spacing)
        iy = int((float(pos[1]) - self.origin[1]) / self.spacing)
        iz = int((float(pos[2]) - self.origin[2]) / self.spacing)
        return ix, iy, iz

    def sample(self, pos):
        ix, iy, iz = self.world_to_grid(pos)
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return float(self.values[ix, iy, iz])
        return 0.0

    def deposit(self, pos, amount):
        ix, iy, iz = self.world_to_grid(pos)
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            self.values[ix, iy, iz] += float(amount)

    def decay(self, dt):
        factor = max(0.0, 1.0 - self.decay_rate * dt)
        self.values *= factor