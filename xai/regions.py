"""Spatial point groups and human-readable part names."""
import numpy as np

PART_ID_NAMES = {
    0: "Body", 1: "Wing", 2: "Tail", 3: "Engine",            # Airplane
    4: "Handle", 5: "Body",                                   # Bag
    6: "Panels", 7: "Peak",                                   # Cap
    8: "Roof", 9: "Hood", 10: "Wheel", 11: "Body",            # Car
    12: "Back", 13: "Seat", 14: "Leg", 15: "Armrest",         # Chair
    16: "Headband", 17: "Earcup", 18: "Cord",                 # Earphone
    19: "Head", 20: "Neck", 21: "Body",                       # Guitar
    22: "Blade", 23: "Handle",                                # Knife
    24: "Base", 25: "Shade", 26: "Canopy", 27: "Pole",        # Lamp
    28: "Keyboard", 29: "Screen",                             # Laptop
    30: "Gas tank", 31: "Seat", 32: "Wheel",                  # Motorbike
    33: "Handle", 34: "Light", 35: "Frame",
    36: "Handle", 37: "Cup",                                  # Mug
    41: "Body", 42: "Fin", 43: "Nose",                        # Rocket
    44: "Wheel", 45: "Deck", 46: "Belt",                      # Skateboard
    47: "Top", 48: "Leg", 49: "Support",                      # Table
}

def build_regions(pts, K, seed):
    """Geometry-only FPS seeds plus nearest-seed assignment; no part labels."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("Expected finite (N,3) points")
    if K < 2 or len(np.unique(pts, axis=0)) < K:
        raise ValueError("Not enough distinct positions for the requested groups")
    seed_idx = np.empty(K, dtype=np.int64)
    seed_idx[0] = np.random.default_rng(seed).integers(0, len(pts))
    dist = np.sum((pts-pts[seed_idx[0]])**2, axis=1)
    for i in range(1, K):
        seed_idx[i] = np.argmax(dist)
        dist = np.minimum(dist, np.sum((pts-pts[seed_idx[i]])**2, axis=1))
    d2 = np.sum((pts[:,None,:]-pts[seed_idx][None,:,:])**2, axis=2)
    labels = d2.argmin(axis=1).astype(np.int64)
    if len(np.unique(labels)) != K:
        raise ValueError("Grouping produced an empty group")
    return labels
