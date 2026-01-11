# Pitch dimensions
PITCH_L = 120
PITCH_W = 80

# === Pitch Thirds (x) ===
X_DEF = PITCH_L / 3
X_ATT = 2 * PITCH_L / 3

# === Lateral Channels (y) ===
Y_LEFT = PITCH_W / 3
Y_CENTRE = 2 * PITCH_W / 3

# === Half-space / Lane System ===
Y_LEFT_WING = PITCH_W * 0.25
Y_LEFT_HALF = PITCH_W * 0.5
Y_RIGHT_HALF = PITCH_W * 0.75
Y_RIGHT_WING = PITCH_W

# === Depth Bands (x) ===
X_DEEP_DEF  = PITCH_L * 0.20
X_BUILD_UP  = PITCH_L * 0.40
X_MIDFIELD  = PITCH_L * 0.60
X_ADV_MID   = PITCH_L * 0.80
X_FINAL_THIRD = PITCH_L

PALETTES = {
    "thirds": "crest",
    "channels": "mako",
    "halfspaces": "flare",
    "depth": "PuBuGn",
    "grid": "crest",
}
