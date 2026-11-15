"""Reproduce TerraScope's geographic generalization audit.

Run from the repository root:
python -m src.analysis.spatial_generalization \
    --validation-h5 data/raw/validation.h5 \
    --testing-h5 data/raw/testing.h5 \
    --validation-geo data/raw/validation_geo.h5 \
    --testing-geo data/raw/testing_geo.h5 \
    --split-path configs/so2sat_split.npz \
    --checkpoint-root /path/to/checkpoints/so2sat \
    --results-root results/so2sat

Coordinates are calculated from the So2Sat tfw metadata. Distances and
footprint intersections are compared only within the same city and EPSG.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.so2sat import So2SatDataset
from src.models.resnet import build_resnet18


CLASS_NAMES = [
    "LCZ1 Compact high-rise", "LCZ2 Compact midrise",
    "LCZ3 Compact low-rise", "LCZ4 Open high-rise",
    "LCZ5 Open midrise", "LCZ6 Open low-rise",
    "LCZ7 Lightweight low-rise", "LCZ8 Large low-rise",
    "LCZ9 Sparsely built", "LCZ10 Heavy industry",
    "LCZA Dense trees", "LCZB Scattered trees",
    "LCZC Bush/scrub", "LCZD Low plants",
    "LCZE Bare rock or paved", "LCZF Bare soil or sand",
    "LCZG Water",
]
NUM_CLASSES = len(CLASS_NAMES)
PATCH_SIZE_M = 320.0


def decode_strings(values):
    result = []
    for value in values.reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            result.append(value.decode("utf-8").strip("\x00 "))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=object)


def read_geo(path):
    with h5py.File(path, "r") as handle:
        cities = decode_strings(handle["city"][:])
        epsg = np.asarray(handle["epsg"][:]).reshape(-1).astype(str)
        tfw = np.asarray(handle["tfw"][:], dtype=np.float64)

    # Pixel-center coordinates for the center of a 32 x 32 patch.
    x = tfw[:, 4] + 15.5 * tfw[:, 0]
    y = tfw[:, 5] + 15.5 * tfw[:, 3]
    return {
        "city": cities,
        "epsg": epsg,
        "x": x,
        "y": y,
    }


def read_labels(path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["label"][:]).argmax(axis=1).astype(np.int64)


def grouped_indices(geo, indices):
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        key = (geo["city"][index], geo["epsg"][index])
        groups.setdefault(key, []).append(int(index))
    return {
        key: np.asarray(values, dtype=np.int64)
        for key, values in groups.items()
    }


def calculate_nearest_distances(train_geo, query_geo, train_indices, query_indices):
    train_groups = grouped_indices(train_geo, train_indices)
    query_groups = grouped_indices(query_geo, query_indices)
    records = []

    for key, query_ids in query_groups.items():
        train_ids = train_groups.get(key)
        if train_ids is None or len(train_ids) == 0:
            raise ValueError(f"No training samples for city/EPSG group {key}")

        train_xy = np.column_stack((
            train_geo["x"][train_ids], train_geo["y"][train_ids]
        ))
        query_xy = np.column_stack((
            query_geo["x"][query_ids], query_geo["y"][query_ids]
        ))
        distances, _ = cKDTree(train_xy).query(query_xy, k=1)

        for index, distance in zip(query_ids, distances):
            records.append({
                "sample_index": int(index),
                "city": key[0],
                "epsg": key[1],
                "nearest_training_distance_m": float(distance),
            })

    return pd.DataFrame(records)


def calculate_footprint_overlap(
    train_geo, query_geo, train_indices, query_indices, partition
):
    train_groups = grouped_indices(train_geo, train_indices)
    query_groups = grouped_indices(query_geo, query_indices)
    records = []

    search_radius = np.sqrt(2.0) * PATCH_SIZE_M

    for key, query_ids in query_groups.items():
        train_ids = train_groups.get(key)
        if train_ids is None or len(train_ids) == 0:
            raise ValueError(f"No training samples for city/EPSG group {key}")

        train_xy = np.column_stack((
            train_geo["x"][train_ids], train_geo["y"][train_ids]
        ))
        query_xy = np.column_stack((
            query_geo["x"][query_ids], query_geo["y"][query_ids]
        ))
        tree = cKDTree(train_xy)

        for sample_index, point in zip(query_ids, query_xy):
            candidates = tree.query_ball_point(point, r=search_radius)
            overlaps = False
            if candidates:
                offsets = np.abs(train_xy[candidates] - point)
                # Positive-area intersection for equal, axis-aligned squares.
                overlaps = bool(np.any(
                    (offsets[:, 0] < PATCH_SIZE_M)
                    & (offsets[:, 1] < PATCH_SIZE_M)
                ))
            records.append({
                "partition": partition,
                "sample_index": int(sample_index),
                "city": key[0],
                "overlaps_training_footprint": overlaps,
            })

    return pd.DataFrame(records)


def distribution_tables(val_labels, test_labels, val_geo, test_geo, out_dir):
    rows = []
    for partition, labels in [
        ("internal_validation", val_labels),
        ("official_test", test_labels),
    ]:
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        for class_id, count in enumerate(counts):
            rows.append({
                "partition": partition,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "count": int(count),
                "proportion": float(count / len(labels)),
            })
    pd.DataFrame(rows).to_csv(out_dir / "class_distribution.csv", index=False)

    city_rows = []
    for partition, geo in [
        ("internal_validation", val_geo),
        ("official_test", test_geo),
    ]:
        cities, counts = np.unique(geo["city"], return_counts=True)
        total = int(counts.sum())
        for city, count in zip(cities, counts):
            city_rows.append({
                "partition": partition,
                "city": city,
                "count": int(count),
                "proportion": float(count / total),
            })
    pd.DataFrame(city_rows).to_csv(
        out_dir / "city_distribution.csv", index=False
    )

    city_class_rows = []
    for city in sorted(set(val_geo["city"]) | set(test_geo["city"])):
        val_mask = val_geo["city"] == city
        test_mask = test_geo["city"] == city
        val_city_labels = val_labels[val_mask]
        test_city_labels = test_labels[test_mask]

        val_counts = np.bincount(val_city_labels, minlength=NUM_CLASSES)
        test_counts = np.bincount(test_city_labels, minlength=NUM_CLASSES)
        val_total = max(1, len(val_city_labels))
        test_total = max(1, len(test_city_labels))

        for class_id in range(NUM_CLASSES):
            val_fraction = val_counts[class_id] / val_total
            test_fraction = test_counts[class_id] / test_total
            city_class_rows.append({
                "city": city,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "validation_count": int(val_counts[class_id]),
                "test_count": int(test_counts[class_id]),
                "validation_proportion": float(val_fraction),
                "test_proportion": float(test_fraction),
                "test_minus_validation_proportion": float(
                    test_fraction - val_fraction
                ),
            })

    pd.DataFrame(city_class_rows).to_csv(
        out_dir / "class_by_city_difference.csv", index=False
    )

    val_counts = np.bincount(val_labels, minlength=NUM_CLASSES)
    test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)
    expected = val_counts * (len(test_labels) / len(val_labels))
    chi2, p_value = chisquare(test_counts, f_exp=expected)

    summary = {
        "class_distribution_chi_square": float(chi2),
        "class_distribution_degrees_of_freedom": NUM_CLASSES - 1,
        "class_distribution_p_value": float(p_value),
        "internal_validation_samples": int(len(val_labels)),
        "official_test_samples": int(len(test_labels)),
    }
    (out_dir / "distribution_test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def load_model(checkpoint_root, experiment, device):
    path = Path(checkpoint_root) / experiment / "best_checkpoint.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_resnet18(NUM_CLASSES, initialization="scratch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def predict_indices(model, h5_path, indices, batch_size, device):
    dataset = So2SatDataset(str(h5_path), indices)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    labels, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(targets.numpy())

    del loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(labels), np.asarray(predictions)


def load_saved_test_predictions(path):
    with np.load(path) as data:
        if "predictions" in data:
            predictions = data["predictions"]
        elif "y_pred" in data:
            predictions = data["y_pred"]
        else:
            raise KeyError(f"No predictions/y_pred array in {path}")
    return np.asarray(predictions, dtype=np.int64)


def per_city_metrics(labels, predictions, geo, model_name, partition):
    rows = []
    for city in sorted(np.unique(geo["city"])):
        mask = geo["city"] == city
        rows.append({
            "model": model_name,
            "partition": partition,
            "city": city,
            "samples": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "macro_f1": float(f1_score(
                labels[mask], predictions[mask],
                labels=np.arange(NUM_CLASSES),
                average="macro", zero_division=0,
            )),
        })
    return rows


def plot_spatial_distribution(
    train_geo, val_geo, test_geo, train_indices, val_indices, out_path
):
    cities = sorted(set(train_geo["city"]) & set(val_geo["city"]) & set(test_geo["city"]))
    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    axes = axes.ravel()

    train_groups = grouped_indices(train_geo, train_indices)
    val_groups = grouped_indices(val_geo, val_indices)
    test_groups = grouped_indices(test_geo, np.arange(len(test_geo["city"])))

    for ax, city in zip(axes, cities):
        # Use a city-local origin so the plot is readable across projections.
        city_train = np.concatenate([
            ids for (name, _), ids in train_groups.items() if name == city
        ])
        city_val = np.concatenate([
            ids for (name, _), ids in val_groups.items() if name == city
        ])
        city_test = np.concatenate([
            ids for (name, _), ids in test_groups.items() if name == city
        ])

        x0 = np.median(train_geo["x"][city_train])
        y0 = np.median(train_geo["y"][city_train])

        ax.scatter(
            (train_geo["x"][city_train] - x0) / 1000,
            (train_geo["y"][city_train] - y0) / 1000,
            s=2, alpha=0.25, label="Training",
        )
        ax.scatter(
            (val_geo["x"][city_val] - x0) / 1000,
            (val_geo["y"][city_val] - y0) / 1000,
            s=4, alpha=0.5, label="Internal validation",
        )
        ax.scatter(
            (test_geo["x"][city_test] - x0) / 1000,
            (test_geo["y"][city_test] - y0) / 1000,
            s=4, alpha=0.5, label="Official test",
        )
        ax.set_title(city)
        ax.set_xlabel("Local easting offset (km)")
        ax.set_ylabel("Local northing offset (km)")
        ax.grid(alpha=0.2)

    for ax in axes[len(cities):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("So2Sat training, validation, and test patch locations")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce TerraScope geographic generalization audit."
    )
    parser.add_argument("--validation-h5", required=True)
    parser.add_argument("--testing-h5", required=True)
    parser.add_argument("--validation-geo", required=True)
    parser.add_argument("--testing-geo", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--results-root", default="results/so2sat")
    parser.add_argument("--models", nargs="+", default=["M4", "M5"],
                        choices=["M4", "M5"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    out_dir = Path(args.results_root) / "distribution_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        device = torch.device(args.device)

    split = np.load(args.split_path)
    train_indices = np.asarray(split["train_idx"], dtype=np.int64)
    val_indices = np.asarray(split["val_idx"], dtype=np.int64)
    test_indices = None

    val_geo = read_geo(args.validation_geo)
    test_geo = read_geo(args.testing_geo)
    train_geo = val_geo

    val_labels_all = read_labels(args.validation_h5)
    test_labels = read_labels(args.testing_h5)
    val_labels = val_labels_all[val_indices]
    test_indices = np.arange(len(test_labels), dtype=np.int64)

    if len(val_geo["city"]) != len(val_labels_all):
        raise ValueError("validation_geo sample count does not match validation HDF5")
    if len(test_geo["city"]) != len(test_labels):
        raise ValueError("testing_geo sample count does not match testing HDF5")

    print("Internal validation:", len(val_indices))
    print("Official test:", len(test_indices))
    print("Device:", device)

    distribution_summary = distribution_tables(
        val_labels, test_labels, val_geo, test_geo, out_dir
    )

    val_distances = calculate_nearest_distances(
        train_geo, val_geo, train_indices, val_indices
    )
    val_distances["partition"] = "internal_validation"
    test_distances = calculate_nearest_distances(
        train_geo, test_geo, train_indices, test_indices
    )
    test_distances["partition"] = "official_test"

    distances = pd.concat(
        [val_distances, test_distances], ignore_index=True
    )
    distances.to_csv(out_dir / "nearest_training_distances.csv", index=False)

    distance_summary = []
    for partition, frame in distances.groupby("partition"):
        values = frame["nearest_training_distance_m"].to_numpy()
        distance_summary.append({
            "partition": partition,
            "samples": int(len(values)),
            "mean_distance_m": float(np.mean(values)),
            "median_distance_m": float(np.median(values)),
            "p90_distance_m": float(np.percentile(values, 90)),
            "within_100m_pct": float(np.mean(values <= 100) * 100),
            "within_250m_pct": float(np.mean(values <= 250) * 100),
            "within_500m_pct": float(np.mean(values <= 500) * 100),
            "within_1km_pct": float(np.mean(values <= 1000) * 100),
        })
    pd.DataFrame(distance_summary).to_csv(
        out_dir / "nearest_training_distance_summary.csv", index=False
    )

    val_overlap = calculate_footprint_overlap(
        train_geo, val_geo, train_indices, val_indices, "internal_validation"
    )
    test_overlap = calculate_footprint_overlap(
        train_geo, test_geo, train_indices, test_indices, "official_test"
    )
    overlap = pd.concat([val_overlap, test_overlap], ignore_index=True)

    overlap_by_city = overlap.groupby(
        ["partition", "city"], as_index=False
    ).agg(
        samples=("overlaps_training_footprint", "size"),
        overlapping_samples=("overlaps_training_footprint", "sum"),
    )
    overlap_by_city["overlap_percent"] = (
        100 * overlap_by_city["overlapping_samples"] / overlap_by_city["samples"]
    )
    overlap_by_city.to_csv(
        out_dir / "training_footprint_overlap_by_city.csv", index=False
    )

    overlap_overall = overlap.groupby("partition", as_index=False).agg(
        samples=("overlaps_training_footprint", "size"),
        overlapping_samples=("overlaps_training_footprint", "sum"),
    )
    overlap_overall["overlap_percent"] = (
        100 * overlap_overall["overlapping_samples"] / overlap_overall["samples"]
    )
    overlap_overall.to_csv(
        out_dir / "training_footprint_overlap_overall.csv", index=False
    )

    per_city_test_rows = []
    validation_test_rows = []

    for model_name in args.models:
        checkpoint = load_model(args.checkpoint_root, model_name, device)

        val_true, val_pred = predict_indices(
            checkpoint, args.validation_h5, val_indices,
            args.batch_size, device,
        )
        test_prediction_path = (
            Path(args.results_root) / model_name / "test_predictions.npz"
        )
        if test_prediction_path.exists():
            test_pred = load_saved_test_predictions(test_prediction_path)
        else:
            _, test_pred = predict_indices(
                checkpoint, args.testing_h5, test_indices,
                args.batch_size, device,
            )

        if len(test_pred) != len(test_labels):
            raise ValueError(
                f"{model_name}: test prediction count does not match test labels"
            )

        val_geo_subset = {
            key: values[val_indices] for key, values in val_geo.items()
        }

        per_city_test_rows.extend(
            per_city_metrics(
                test_labels, test_pred, test_geo, model_name, "official_test"
            )
        )
        val_rows = per_city_metrics(
            val_true, val_pred, val_geo_subset, model_name, "internal_validation"
        )
        val_by_city = {row["city"]: row for row in val_rows}
        test_rows = {
            row["city"]: row
            for row in per_city_metrics(
                test_labels, test_pred, test_geo, model_name, "official_test"
            )
        }

        for city in sorted(val_by_city):
            v = val_by_city[city]
            t = test_rows[city]
            validation_test_rows.append({
                "model": model_name,
                "city": city,
                "validation_samples": v["samples"],
                "validation_accuracy": v["accuracy"],
                "validation_macro_f1": v["macro_f1"],
                "test_samples": t["samples"],
                "test_accuracy": t["accuracy"],
                "test_macro_f1": t["macro_f1"],
                "accuracy_change_percentage_points": (
                    t["accuracy"] - v["accuracy"]
                ) * 100,
            })

        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_metrics_df = pd.DataFrame(per_city_test_rows)
    test_metrics_df.to_csv(
        out_dir / "per_city_test_metrics.csv", index=False
    )
    test_metrics_df.to_csv(
        out_dir / "per_city_test_metrics_corrected.csv", index=False
    )
    pd.DataFrame(validation_test_rows).to_csv(
        out_dir / "validation_vs_test_per_city.csv", index=False
    )

    plot_spatial_distribution(
        train_geo, val_geo, test_geo,
        train_indices, val_indices,
        out_dir / "spatial_distribution_by_city.png",
    )

    overlap_summary = {
        row["partition"]: round(float(row["overlap_percent"]), 3)
        for _, row in overlap_overall.iterrows()
    }
    final_summary = {
        **distribution_summary,
        "training_footprint_overlap_percent": overlap_summary,
        "patch_dimensions_pixels": [32, 32],
        "pixel_resolution_m": 10,
        "patch_footprint_m": [320, 320],
        "interpretation": (
            "Spatial association differs between the random internal "
            "validation split and the official test partition. This "
            "observational audit does not establish causation."
        ),
    }
    (out_dir / "spatial_audit_summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n"
    )

    print("\nSpatial audit complete.")
    print(json.dumps(final_summary, indent=2))
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()
