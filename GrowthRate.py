import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

DEFAULT_EXCEL_FILE = "data.xlsx"
OUTPUT_DIR = Path.home() / "Desktop" / "Experiment" / "Result"

Label = np.array([
"Stdt", "Scmr", "Syfp", "Snp", "Ctdt", "Cyfp", "Cnp", "Rtdt", "Rcmr", "Ryfp", "Rnp"
])

Y_NAMES = np.array(["S_tdt_1", "S_cmr_1", "C_tdt_1", "R_cmr_1", "R_tdt_1", "C_cmr_1", "S_tdtR_yfp_1", "R_tdtS_yfp_1",
            "S_tdt_2", "S_cmr_2", "C_tdt_2", "R_cmr_2", "R_tdt_2", "C_cmr_2", "S_tdtR_yfp_2", "R_tdtS_yfp_2",
            "S_tdt_3", "S_cmr_3", "C_tdt_3", "R_cmr_3", "R_tdt_3", "C_cmr_3", "S_tdtC_yfp", "C_tdtS_yfp",
            "S_yfp_1", "S_np_1", "C_yfp_1", "C_np_1", "R_yfp_1", "R_np_1", "R_tdtC_yfp", "C_tdtR_yfp",
            "S_yfp_2", "S_np_2", "C_yfp_2", "C_np_2", "R_yfp_2", "R_np_2", "S_tdtS_yfp", "C_tdtC_yfp",
            "S_yfp_3", "S_np_3", "C_yfp_3", "C_np_3", "R_yfp_3", "R_np_3", "R_tdtR_yfp"
            ])

RANGES = np.array([(None,None), (None,None), (17,26), (8,18), (9,19), (None, None), (None, None), (None, None),
                    (8,20), (8,21), (15,24), (12,22), (8,20), (None, None), (None, None), (None, None),
                    (11,20), (6,18), (None,None), (16,22), (9,22), (None, None), (None, None), (None, None),
                    (6,15), (4,11), (16,26), (5,10), (6,14), (1,12), (None, None), (None, None),
                    (6,13), (6,12), (10,17), (6,12), (7,17), (5,12), (None, None), (None, None),
                    (5,14), (None,None), (None,None), (None,None), (4,16), (4,12), (None, None)]
                    ,dtype=object)

def processing(df):
    x = df.iloc[1,1:].to_numpy(dtype=float)
    y = (df.iloc[3:50,1:] - df.iloc[50,1:]).to_numpy(dtype=float)

    if len(x) < 3:
        raise ValueError("Numerical data should be at least 3 points")

    if np.min(x) == np.max(x):
        raise ValueError("All x are same values")
    return x, y

def lsm_fit_with_ranges(x,y, ranges=RANGES):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n_series, n_points = y.shape

    bounds = pd.DataFrame(ranges, columns=["start", "end"])
 
    present = bounds.notna().to_numpy()
  
    active = present.all(axis=1)

    numeric_bounds = bounds.apply(pd.to_numeric, errors="raise").fillna(0).to_numpy(dtype=float)

    start, end = numeric_bounds.T

    positive = np.isfinite(x)[None, :] & np.isfinite(y) & (y > 0)

    start = np.maximum(start, 1).astype(int)
    end = end.astype(int)

    log_y = np.full_like(y, np.nan)
    np.log(y,out=log_y, where=(y>0) & np.isfinite(y))

    visible_number = np.cumsum(positive, axis=1)
    mask = (
        active[:, None]
        & positive
        & (visible_number >= start[:, None])
        & (visible_number <= end[:, None])
    )

    n = mask.sum(axis=1)

    x_mean = np.where(mask, x[None, :], 0).sum(axis=1) / np.maximum(n,1)
    log_y_mean = np.where(mask, log_y, 0).sum(axis=1) / np.maximum(n,1)

    dx = np.where(mask, x[None,:] - x_mean[:,None], 0)
    dy = np.where(mask, log_y - log_y_mean[:,None], 0)

    sxx = np.sum(dx**2, axis=1)
    sxy = np.sum(dx*dy, axis=1)
    syy = np.sum(dy**2, axis=1)
    valid = active & (n>=3) & (sxx>0)

    a = np.full(n_series, np.nan)
    np.divide(sxy, sxx, out=a, where=valid)
    b = np.where(valid, log_y_mean - a*x_mean, np.nan)

    r_squared = np.full(n_series, np.nan)
    np.divide(sxy**2, sxx*syy, out=r_squared, where=valid & (syy>0))

    return a, b, r_squared, mask

def make_figs(x, y, y_names, a, b, r_squared, mask, output_dir=OUTPUT_DIR):

    if len(y_names) != len(y):
        raise ValueError("The number of series names must match the number of series.")
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(y_names):
        fig, ax = plt.subplots(figsize=(8,5))
        positive = np.isfinite(x) & np.isfinite(y[i]) & (y[i]>0)
        ax.scatter(x[positive], y[i,positive], s=40, alpha=0.8, label=f"Data (n={positive.sum()})")

        if np.isfinite(a[i]) and np.isfinite(b[i]):
            fit_x = x[mask[i]]
            ax.scatter(fit_x, y[i, mask[i]], s=70, facecolors="none", 
                       edgecolors= "tab:orange", label=f"Used for fit (n={fit_x.size})")

            line_x = np.linspace(fit_x.min(), fit_x.max(), 200)
            line_y = np.exp(a[i] * line_x + b[i])

            ax.plot(line_x, line_y, color="tab:red", label=f"Fit: y= exp({a[i]:.4g}t {b[i]:+.4g})")
            r2_text = f"{r_squared[i]:.4f}" if np.isfinite(r_squared[i]) else "undefined"
            visible_number = np.cumsum(positive)
            used_numbers = visible_number[mask[i]]
            ax.text(0.03, 0.97,
                    f"R^2 (log y) = {r2_text}\n"
                    f"Visible points: {used_numbers[0]} to {used_numbers[-1]} (1-based)",
                    transform=ax.transAxes, va="top", fontsize=9)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ln(OD/OD_blank)")
        ax.set_yscale("log")
        ax.set_title(f"Exponential Fit of {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        filename = f"fitting_{name}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

def figs_by_strain(x, y, y_names, a, b, r_squared, mask, output_dir=OUTPUT_DIR):
    strain_groups = {
        "Stdt": [0, 8, 16],
        "Scmr": [1, 9, 17],
        "Ctdt": [2, 10, 18],
        "Rcmr": [3, 11, 19],
        "Rtdt": [4, 12, 20],
        "Syfp": [24, 32, 40],
        "Snp": [25, 33, 41],
        "Cyfp": [26, 34, 42],
        "Cnp": [27, 35, 43],
        "Ryfp": [28, 36, 44],
        "Rnp": [29, 37, 45]
    }

    for strain_name, indices in strain_groups.items():
        fig, ax = plt.subplots(figsize=(8,5))

        colors = ["orange", "green", "blue"]

        for i in indices:
            positive = np.isfinite(x) & np.isfinite(y[i]) & (y[i]>0)
            ax.scatter(x[positive], y[i,positive], s=40, alpha=0.8,
                       label=f"{y_names[i]} (n={positive.sum()})")

            if np.isfinite(a[i]) and np.isfinite(b[i]):
                fit_x = x[mask[i]]
                ax.scatter(fit_x, y[i, mask[i]], s=70, facecolors="none", 
                           edgecolors= colors[i % 3], label=f"Used for fit {y_names[i]} (n={fit_x.size})")

                line_x = np.linspace(fit_x.min(), fit_x.max(), 200)
                line_y = np.exp(a[i] * line_x + b[i])
                ax.plot(line_x, line_y, label=f"Fit {y_names[i]}: y= exp({a[i]:.4g}t {b[i]:+.4g})", color=colors[i % 3])

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ln(OD/OD_blank)")
        ax.set_yscale("log")
        ax.set_title(f"Exponential Fit of {strain_name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        filename = f"fitting_{strain_name}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

def main() -> None:
    excel_file = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXCEL_FILE)

    df = pd.read_excel(excel_file, header=None, index_col=None)
    x, y = processing(df)

    a, b, r_squared, mask = lsm_fit_with_ranges(x, y)

    make_figs(x, y, Y_NAMES, a, b, r_squared, mask)
    figs_by_strain(x, y, Y_NAMES, a, b, r_squared, mask)
    print(f"Figures saved to: {OUTPUT_DIR}")

    print("a:", a)
    print("b:", b)
    
    Stdt = (a[8] + a[16]) / 2
    Scmr = (a[9] + a[17]) / 2
    Ctdt = (a[2] + a[10]) / 2
    Rcmr = (a[3] + a[11] + a[19]) / 3
    Rtdt = (a[4] + a[12] + a[20]) / 3
    Syfp = (a[24] + a[32] + a[40]) / 3
    Snp = (a[25] + a[33]) / 2
    Cyfp = (a[26] + a[34]) / 2
    Cnp = (a[27] + a[35]) / 2
    Ryfp = (a[28] + a[36] + a[44]) / 3
    Rnp = (a[29] + a[37] + a[45]) / 3
    
    Mean = np.array([Stdt, Scmr, Syfp, Snp, Ctdt, Cyfp, Cnp, Rtdt, Rcmr, Ryfp, Rnp])

    Stdt_var = ((a[8] - Stdt)**2 + (a[16] - Stdt)**2) / 2
    Scmr_var = ((a[9] - Scmr)**2 + (a[17] - Scmr)**2) / 2
    Ctdt_var = ((a[2] - Ctdt)**2 + (a[10] - Ctdt)**2) / 2
    Rcmr_var = ((a[3] - Rcmr)**2 + (a[11] - Rcmr)**2 + (a[19] - Rcmr)**2) / 3
    Rtdt_var = ((a[4] - Rtdt)**2 + (a[12] - Rtdt)**2 + (a[20] - Rtdt)**2) / 3
    Syfp_var = ((a[24] - Syfp)**2 + (a[32] - Syfp)**2 + (a[40] - Syfp)**2) / 3
    Snp_var = ((a[25] - Snp)**2 + (a[33] - Snp)**2) / 2
    Cyfp_var = ((a[26] - Cyfp)**2 + (a[34] - Cyfp)**2) / 2
    Cnp_var = ((a[27] - Cnp)**2 + (a[35] - Cnp)**2) / 2
    Ryfp_var = ((a[28] - Ryfp)**2 + (a[36] - Ryfp)**2 + (a[44] - Ryfp)**2) / 3
    Rnp_var = ((a[29] - Rnp)**2 + (a[37] - Rnp)**2 + (a[45] - Rnp)**2) / 3
    
    Std = np.sqrt(np.array([Stdt_var, Scmr_var, Syfp_var, Snp_var, Ctdt_var, Cyfp_var, Cnp_var, Rtdt_var, Rcmr_var, Ryfp_var, Rnp_var]))

    Dbt = np.log(2) / Mean
    Dbt_std = Std * np.log(2) / Mean**2
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8,5))
    ax1.errorbar(Label, Mean, yerr = Std, capsize=5, fmt='o', markersize=5, ecolor='blue', markeredgecolor = "blue", color='b')
    ax1.set_xlabel("Strain", fontsize = 24)
    ax1.set_ylabel("Growth Rate [1/s]", fontsize = 24)

    ax2.errorbar(Label, Dbt, yerr = Dbt_std, capsize=5, fmt='o', markersize=5, ecolor='blue', markeredgecolor = "blue", color='b')
    ax2.set_xlabel("Strain", fontsize = 24)
    ax2.set_ylabel("Doubling Time [s]", fontsize = 24)

    fig.savefig(OUTPUT_DIR / "Results.png", dpi=300, bbox_inches="tight")
    
if __name__ == "__main__":
    main()