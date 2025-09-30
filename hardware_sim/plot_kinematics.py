
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main(filename):
    # Read CSV file without headers, strip spaces
    df = pd.read_csv(filename, header=None, skipinitialspace=True)

    # Keep only the first 10 columns (ignore iterations, duration)
    df = df.iloc[:, :10]

    # Assign column names
    df.columns = [
        "time",
        "y",
        "y_est",
        "y_measured",
        "v",
        "v_est",
        "a",
        "a_est",
        "a_measured",
        "apogee_est"
    ]

    # Convert all columns to numeric, invalid values become NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows that contain any NaNs
    nans = df[df.isna().any(axis=1)]
    if not nans.empty:
        print(f"Skipping {len(nans)} malformed rows:")
        print(nans)
    df = df.dropna()

    print("Loaded data with shape:", df.shape)
    print(df.head())

    # True apogee (max of measured altitude y_measured)
    true_apogee = df["y_measured"].max()

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Position
    axs[0].plot(df["time"], df["y"], label="y (sensor)")
    axs[0].plot(df["time"], df["y_est"], label="y_est (KF estimate)")
    axs[0].plot(df["time"], df["y_measured"], label="y_measured (ground truth)", linestyle=":")
    axs[0].axhline(true_apogee, color="red", linestyle="--", label=f"True Apogee = {true_apogee:.2f}")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    # Velocity
    axs[1].plot(df["time"], df["v"], label="v (sensor)")
    axs[1].plot(df["time"], df["v_est"], label="v_est (KF estimate)")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    # Acceleration
    axs[2].plot(df["time"], df["a"], label="a (sensor)")
    axs[2].plot(df["time"], df["a_est"], label="a_est (KF estimate)")
    axs[2].plot(df["time"], df["a_measured"], label="a_measured (ground truth)", linestyle=":")
    axs[2].set_ylabel("Acceleration")
    axs[2].legend()

    # Apogee estimate vs true apogee
    axs[3].plot(df["time"], df["apogee_est"], color="purple", label="Apogee Estimate")
    axs[3].axhline(true_apogee, color="red", linestyle="--", label=f"True Apogee = {true_apogee:.2f}")
    axs[3].set_ylabel("Apogee")
    axs[3].set_xlabel("Time")
    axs[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} data.csv")
        sys.exit(1)
    main(sys.argv[1])

