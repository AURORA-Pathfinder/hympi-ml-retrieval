"""
Microwave Channel Grouping CLI
Combines 50–58 GHz and 175–191 GHz grouping logic into one script.

python grouping_cli.py \
    --input /path/to/data.nc \
    --mode 50 \
    --output-dir ./results \
    --test-name mytest \
    --alpha 500

"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# -------------------- Data I/O --------------------

def read_netcdf(filename):
    with nc.Dataset(filename) as ncf:
        freq = ncf.variables['FREQUENCY'][:]
        tb = ncf.variables['BT'][:]
    return freq, tb


# -------------------- Clustering --------------------

def run_cluster(data_tb, indices, threshold, neighborhood, nedt=None):
    sub_data = data_tb[indices, :]
    if nedt is not None:
        sub_data = sub_data / nedt
        sub_data = sub_data - sub_data.mean(axis=0)

    corr = np.corrcoef(sub_data)
    dist = 1 - np.abs(corr)

    conn = lil_matrix((len(indices), len(indices)))
    for i in range(len(indices)):
        for j in range(1, neighborhood + 1):
            if i - j >= 0:
                conn[i, i - j] = 1
            if i + j < len(indices):
                conn[i, i + j] = 1

    try:
        # For scikit-learn >=1.2
        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            connectivity=conn,
            distance_threshold=threshold,
            n_clusters=None
        )
    except TypeError:
        # Fallback for older scikit-learn
        clustering = AgglomerativeClustering(
            affinity='precomputed',
            linkage='average',
            connectivity=conn,
            distance_threshold=threshold,
            n_clusters=None
        )

    return clustering.fit_predict(dist)


# -------------------- Grouping --------------------

def compute_center_bandwidth(n_groups, frequencies, labels, tb):
    centers, bandwidths, counts, freqs, tbs = [], [], [], [], []
    spacing = frequencies[1] - frequencies[0]

    for gid in range(n_groups):
        group_freq = frequencies[labels == gid]
        group_tb = tb[labels == gid, :]
        cnt = len(group_freq)
        if cnt == 1:
            center, bandwidth = group_freq[0], spacing
        else:
            center = group_freq[cnt // 2] if cnt % 2 else group_freq[cnt // 2 - 1]
            bandwidth = group_freq[-1] - group_freq[0]
        centers.append(center)
        bandwidths.append(bandwidth)
        counts.append(cnt)
        tbs.append(group_tb)
        freqs.append(group_freq)

    sorted_data = sorted(zip(centers, bandwidths, counts, tbs, freqs), key=lambda x: x[0])
    c, b, ct, tb_sorted, fq = zip(*sorted_data)
    return list(c), list(b), list(ct), list(fq), list(tb_sorted)


# -------------------- Processing --------------------

class GroupedSRFProcessor:
    def __init__(self, groups):
        self.groups = groups
        self.center_freqs, self.avg_tbs = [], []

    @staticmethod
    def trapz(values, x):
        return np.trapz(values, x)

    def process(self):
        for group in self.groups:
            freqs = np.array(group['freqs'])
            tbs = np.array(group['tbs'])
            bw, cf = group['bandwidth'], group['center_freq']
            if tbs.ndim == 1:
                tbs = tbs[:, np.newaxis]
            mask = (freqs >= cf - bw / 2) & (freqs <= cf + bw / 2)
            sel_freqs, sel_tbs = freqs[mask], tbs[mask, :]
            if len(sel_freqs) < 2:
                avg_tb = sel_tbs[0, :]
            else:
                numerator = np.array([self.trapz(sel_tbs[:, p], sel_freqs) for p in range(tbs.shape[1])])
                denominator = self.trapz(np.ones_like(sel_freqs), sel_freqs)
                avg_tb = numerator / denominator if denominator != 0 else np.full(tbs.shape[1], np.nan)
            self.center_freqs.append(cf)
            self.avg_tbs.append(avg_tb)


def reconstruct_regression(compressed_data, data_tb, alpha=1.0):
    X_train, _, y_train, _ = train_test_split(compressed_data, data_tb, test_size=0.2, random_state=42)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model.predict(compressed_data)


# -------------------- Stats & Plots --------------------

def compute_stats(tb_all, tb_new, group_centers, nedt_all, tau, t_rec):
    error_bias = np.mean(tb_all - tb_new, axis=0)
    error_std = np.std(tb_all - tb_new, axis=0)
    tb_avg = np.mean(tb_all, axis=0)
    tb_new_avg = np.mean(tb_new, axis=0)
    t_sys = 290 + t_rec
    group_freq = np.array(group_centers)
    bsub = np.diff(group_freq)
    bsub = np.append(bsub, bsub[-1]) * 1e9
    nedt_sub = t_sys / (np.sqrt(bsub * tau))
    return error_bias, error_std, tb_new_avg, tb_avg, nedt_sub


def bar_plot(img_name, centers, counts, bandwidths):
    plt.figure(figsize=(12, 4))
    plt.bar(centers, counts, width=bandwidths, align='center', edgecolor='k')
    plt.xlabel("Center Frequency (GHz)")
    plt.ylabel("Channel Count")
    plt.title("Grouped Channel Bandwidths")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)


# -------------------- Mode-specific configs --------------------

def get_mode_config(mode):
    if mode == "50":
        return {
            "nedt": 3.74,
            "t_rec": 450,
            "segments": [
                (lambda f: f <= 51, 1e-6, 5),
                (lambda f: (f > 51) & (f < 54), 1e-5, 2),
                (lambda f: f >= 54, 1e-4, 5)
            ]
        }
    elif mode == "183":
        return {
            "nedt": 6.4,
            "t_rec": 1000,
            "segments": [
                (lambda f: f < 178, 1e-5, 5),
                (lambda f: (f >= 178) & (f < 182), 1e-3, 5),
                (lambda f: (f >= 182) & (f < 184), 1e-4, 2),
                (lambda f: (f >= 184) & (f < 188), 1e-3, 5),
                (lambda f: f >= 188, 1e-4, 5)
            ]
        }
    else:
        raise ValueError("Unsupported mode. Use '50' or '183'.")


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Microwave Channel Grouping")
    parser.add_argument("--input", required=True, help="Input NetCDF file")
    parser.add_argument("--mode", required=True, choices=["50", "183"], help="Grouping mode: 50 or 183 GHz")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--test-name", default="test", help="Test name for outputs")
    parser.add_argument("--alpha", type=float, default=1000.0, help="Ridge regression alpha")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = get_mode_config(args.mode)
    nedt_val, t_rec, segments = cfg["nedt"], cfg["t_rec"], cfg["segments"]

    frequencies, tb_all = read_netcdf(args.input)
    data_tb = tb_all.T
    n_channels = data_tb.shape[0]
    labels = np.full(n_channels, -1, dtype=int)

    # Cluster segments
    offset = 0
    for mask_func, thresh, neigh in segments:
        idx = np.where(mask_func(frequencies))[0]
        seg_labels = run_cluster(data_tb, idx, thresh, neigh, nedt=nedt_val)
        labels[idx] = seg_labels + offset
        offset = labels[idx].max() + 1
    n_groups = labels.max() + 1

    # Group processing
    centers, bandwidths, counts, freqs, tbs = compute_center_bandwidth(n_groups, frequencies, labels, data_tb)
    groups = [{"freqs": fq.tolist(), "tbs": tb.tolist(), "bandwidth": bw, "center_freq": cf}
              for fq, tb, bw, cf in zip(freqs, tbs, bandwidths, centers)]
    processor = GroupedSRFProcessor(groups)
    processor.process()
    tb_filtered = np.vstack(processor.avg_tbs).T
    f_filtered = np.array(processor.center_freqs)

    # Reconstruct
    tb_recon = reconstruct_regression(tb_filtered, tb_all, alpha=args.alpha)

    # Stats
    error_bias, error_std, tb_new_avg, tb_avg, nedt_sub = compute_stats(
        tb_all, tb_recon, centers, nedt_val, 0.01, t_rec
    )

    # Save results
    np.savez(os.path.join(args.output_dir, f"{args.test_name}_grouped_TB_{args.mode}.npz"),
             TB=tb_filtered, freq=f_filtered, bandwidth=bandwidths)

    # Bar plot
    bar_plot(os.path.join(args.output_dir, f"{args.test_name}_bar_{args.mode}.png"),
             centers, counts, bandwidths)

    print(f"Completed. Groups: {n_groups}. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
