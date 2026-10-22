"""Sample every power-monitor channel from sysfs (docs/02_protocol.md §4) into a tidy CSV: t_s, rail, mW, mV, mA.

Channels are found by label, never by hwmon index (indices change across boots):
  ina3221 (hwmon name "ina3221"): inN_label -> rail; bus voltage inN_input (mV), current currN_input (mA),
          mW = mV * mA / 1000 (derived).
  ina238  (hwmon name "ina238"):  label -> rail; in1_input (bus mV), curr1_input (mA), power1_input (uW, measured).
Also writes the channel table (rail, chip, sysfs paths) next to the CSV.

python bench/power_sample.py --seconds 10 --hz 10 --out results/<run>/raw/power_idle.csv
"""
import argparse
import csv
import glob
import os
import time


def rd(path):
    with open(path) as f:
        return f.read().strip()


def find_channels():
    chans = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = rd(f"{h}/name")
        if name == "ina3221":
            for lab in sorted(glob.glob(f"{h}/in*_label")):
                n = os.path.basename(lab).split("_")[0][2:]
                rail = rd(lab)
                if os.path.exists(f"{h}/curr{n}_input"):
                    chans.append({"rail": rail, "chip": name, "mv": f"{h}/in{n}_input", "ma": f"{h}/curr{n}_input",
                                  "uw": ""})
        elif name == "ina238":
            rail = rd(f"{h}/label") if os.path.exists(f"{h}/label") else "ina238"
            chans.append({"rail": rail, "chip": name, "mv": f"{h}/in1_input", "ma": f"{h}/curr1_input",
                          "uw": f"{h}/power1_input"})
    return chans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chans = find_channels()
    with open(a.out.replace(".csv", "_channels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rail", "chip", "mv", "ma", "uw"])
        w.writeheader()
        w.writerows(chans)
    period = 1.0 / a.hz
    t0 = time.monotonic()
    rows = []
    k = 0
    while True:
        t = time.monotonic() - t0
        if t >= a.seconds:
            break
        for c in chans:
            mv = float(rd(c["mv"]))
            ma = float(rd(c["ma"]))
            mw = float(rd(c["uw"])) / 1000.0 if c["uw"] else mv * ma / 1000.0
            rows.append({"t_s": round(t, 4), "rail": c["rail"], "mW": round(mw, 1), "mV": mv, "mA": ma})
        k += 1
        time.sleep(max(0.0, t0 + k * period - time.monotonic()))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "rail", "mW", "mV", "mA"])
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["rail"], []).append(r["mW"])
    for rail, v in by.items():
        print(f"{rail:20s} mean {sum(v) / len(v):9.1f} mW  n={len(v)}")


if __name__ == "__main__":
    main()
