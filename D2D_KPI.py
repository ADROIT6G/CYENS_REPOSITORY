#!/usr/bin/env python3
import math
import pandas as pd
import numpy as np

# Constants
CSV_FILE = "topology-data-single-run-20UEs.csv"
PATH_EXP = 3
EARTH_R = 6_371_000
TEMP_K = 290
BOLTZ = 1.38064852e-23
C_LIGHT = 299792458

# Antenna gains (linear scale)
BS_RX_GAIN_dBi = 14
UE_TX_GAIN_dBi = 0
PARENT_RX_GAIN_dBi = 6
BS_RX_GAIN = 10 ** (BS_RX_GAIN_dBi / 10)
UE_TX_GAIN = 10 ** (UE_TX_GAIN_dBi / 10)
PARENT_RX_GAIN = 10 ** (PARENT_RX_GAIN_dBi / 10)

# Max BW and MIMO setup
MIMO_STREAMS = 4
NR_BANDS = {
    "n77":  {"fc": 3.7e9,  "bw": 200e6,  "eirp_dbm": 30},
    "n260": {"fc": 39e9,  "bw": 1600e6, "eirp_dbm": 30},
    "n261": {"fc": 28e9,  "bw": 1600e6, "eirp_dbm": 30}
}

# Helper functions
def haversine_m(row, bs_lat, bs_lon):
    phi1, phi2 = math.radians(row["Latitude"]), math.radians(bs_lat)
    dphi = math.radians(bs_lat - row["Latitude"])
    dlmb = math.radians(bs_lon - row["Longitude"])
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return EARTH_R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def dbm_to_w(dbm):
    return 10 ** ((dbm - 30) / 10)

def noise(bw):
    return np.random.normal(loc=BOLTZ * TEMP_K * bw, scale=1e-18)

def rayleigh_path_loss(fc, d):
    lambda_c = C_LIGHT / fc
    return (d ** PATH_EXP) / (lambda_c ** 2)

def snr_rayleigh(pt, d, fc, g_tx, g_rx, bw):
    if d == 0: d = 0.1
    path_loss = rayleigh_path_loss(fc, d)
    prx = pt * g_tx * g_rx / path_loss
    return prx / abs(noise(bw))

def capacity(bw, snr_lin):
    return bw * math.log2(1 + snr_lin)

def fmt_power(p):
    if p < 1e3: return f"{p:.1f} mW"
    return f"{p / 1e3:.1f} W"

# Load data
df = pd.read_csv(CSV_FILE)
bs = df[df["Name"] == "BaseStation"].iloc[0]
ues = df[df["Name"] != "BaseStation"].copy()

# Simulation
results = []

for tag, band in NR_BANDS.items():
    fc, bw, eirp_dbm = band["fc"], band["bw"], band["eirp_dbm"]
    pt = dbm_to_w(eirp_dbm)

    power_baseline = 0
    power_d2d = 0
    rate_baseline = 0
    rate_d2d = 0
    parents = set()

    for _, ue in ues.iterrows():
        d_bs = haversine_m(ue, bs["Latitude"], bs["Longitude"])
        snr_lin = max(snr_rayleigh(pt, d_bs, fc, UE_TX_GAIN, BS_RX_GAIN, bw), 0.01)
        rate_baseline += MIMO_STREAMS * capacity(bw, snr_lin)
        power_baseline += (d_bs / 1000) ** PATH_EXP

    for _, ue in ues.iterrows():
        parent = bs if pd.isna(ue["Parent ID"]) else df[df["ID"] == ue["Parent ID"]].iloc[0]
        d_up = haversine_m(ue, parent["Latitude"], parent["Longitude"])
        g_rx = BS_RX_GAIN if parent["ID"] == bs["ID"] else PARENT_RX_GAIN
        snr_lin_up = max(snr_rayleigh(pt, d_up, fc, UE_TX_GAIN, g_rx, bw), 0.01)
        rate_d2d += MIMO_STREAMS * capacity(bw, snr_lin_up)
        power_d2d += (d_up / 1000) ** PATH_EXP
        if parent["ID"] != bs["ID"]:
            parents.add(parent["ID"])

    for pid in parents:
        parent = df[df["ID"] == pid].iloc[0]
        d_pb = haversine_m(parent, bs["Latitude"], bs["Longitude"])
        snr_lin_pb = max(snr_rayleigh(pt, d_pb, fc, UE_TX_GAIN, BS_RX_GAIN, bw), 0.01)
        rate_d2d += MIMO_STREAMS * capacity(bw, snr_lin_pb)
        power_d2d += (d_pb / 1000) ** PATH_EXP

    power_saving = (1 - power_d2d / power_baseline) * 100
    rate_gain = (rate_d2d - rate_baseline) / rate_baseline * 100

    results.append({
        "Band": tag,
        "Frequency (GHz)": f"{fc / 1e9:.1f}",
        "Baseline Power": fmt_power(power_baseline),
        "D2D Power": fmt_power(power_d2d),
        "Power Savings (%)": f"{power_saving:.1f}%",
        "Baseline Rate": f"{rate_baseline / 1e9:.2f} Gbps",
        "D2D Rate": f"{rate_d2d / 1e9:.2f} Gbps",
        "Sum Rate Gain (%)": f"{rate_gain:.1f}%"
    })

# Output results
df_final = pd.DataFrame(results)
print("\n=== 5G Max BW + 4x4 MIMO Simulation ===")
print(df_final.to_string(index=False))
