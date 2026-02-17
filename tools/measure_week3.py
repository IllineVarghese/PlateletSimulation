import numpy as np

L = 10.0
R = 1.0
near_band = 0.10 * R
act_threshold = 0.02

P = np.load("results/week3/positions_saved.npy")
A = np.load("results/week3/activation_saved.npy")

p0 = P[-2]
p1 = P[-1]
a = A[-1]

dz = (p1[:, 2] - p0[:, 2]) % L
r = np.sqrt(p0[:, 0] ** 2 + p0[:, 1] ** 2)

near = (R - r) <= near_band
sel = near & (a >= act_threshold)

print("near count:", int(near.sum()))
print("sel count:", int(sel.sum()))
print("dz near all:", float(dz[near].mean()))
print("dz sel (eligible):", float(dz[sel].mean()) if sel.sum() > 0 else None)
print("A min/mean/max:", float(a.min()), float(a.mean()), float(a.max()))

far = ~near

print("dz far:", float(dz[far].mean()) if far.sum() > 0 else None)
print("A mean near:", float(a[near].mean()) if near.sum() > 0 else None)
print("A mean far :", float(a[far].mean()) if far.sum() > 0 else None)
