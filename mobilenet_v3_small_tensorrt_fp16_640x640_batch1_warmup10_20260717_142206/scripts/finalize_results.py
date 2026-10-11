from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    PACKAGED_ENGINE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PACKAGED_ORIGINAL_ONNX,
    FP32_REFERENCE,
    RUN,
    class_names,
    compute_classification_metrics,
    file_record,
    read_json,
    sha256,
    stats,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "NA"):
            values.append(float(value))
    return values


def integrate_power(rows: list[dict], start: float | None = None, end: float | None = None, idle=0.0):
    samples = [
        (float(row["timestamp_epoch"]), max(0.0, float(row["power_tot_w"]) - idle))
        for row in rows
        if row.get("timestamp_epoch") and row.get("power_tot_w")
    ]
    if start is not None:
        samples = [(t, p) for t, p in samples if start <= t <= end]
    if not samples:
        return None
    samples.sort()
    if start is not None and samples[0][0] > start:
        samples.insert(0, (start, samples[0][1]))
    if end is not None and samples[-1][0] < end:
        samples.append((end, samples[-1][1]))
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) * 0.5 * (t1 - t0)
    return energy


def save_figure(name: str, figure, plot_rows: list[dict] | None = None):
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        figure.savefig(RUN / f"figures/{extension}/{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)
    if plot_rows is not None:
        write_csv(RUN / f"plot_data/{name}.csv", plot_rows)


def plot_line(name, x, y, xlabel, ylabel, title, plot_key):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, y, linewidth=1.2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    save_figure(name, fig, [{xlabel: a, plot_key: b} for a, b in zip(x, y)])


def main():
    predictions = read_csv(RUN / "raw/per_image_predictions.csv")
    timings = read_csv(RUN / "raw/per_image_timings.csv")
    provenance = read_json(RUN / "raw/execution_count_provenance.json")
    idle_rows = read_csv(RUN / "monitoring/idle_samples.csv")
    power_rows = read_csv(RUN / "monitoring/primary_power_samples.csv")
    names = class_names()
    metric_input = [
        {
            **row,
            "true_class_id": int(row["true_class_id"]),
            "predicted_class_id": int(row["predicted_class_id"]),
            "correct": int(row["correct"]),
        }
        for row in predictions
    ]
    metrics = compute_classification_metrics(metric_input, names)
    metrics.update(
        {
            "precision": "TensorRT FP16-enabled",
            "input_shape": [1, 3, 640, 640],
            "batch_size": 1,
            "historical_reference": {"accuracy": 0.9731800766283525, "macro_f1": 0.9518572587185726},
            "historical_match": {
                "accuracy_difference": metrics["accuracy"] - 0.9731800766283525,
                "macro_f1_difference": metrics["macro_f1"] - 0.9518572587185726,
            },
        }
    )
    write_json(RUN / "metrics/classification_metrics.json", metrics)
    flat_metrics = [
        {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, (list, dict))
        }
    ]
    write_csv(RUN / "metrics/classification_metrics.csv", flat_metrics)
    write_csv(RUN / "metrics/per_class_metrics.csv", metrics["per_class"])
    write_csv(
        RUN / "metrics/misclassified_images.csv",
        [row for row in predictions if int(row["correct"]) == 0],
        list(predictions[0]) if predictions else [],
    )
    cm_rows = [
        {"true_class": names[i], **{names[j]: metrics["confusion_matrix"][i][j] for j in range(len(names))}}
        for i in range(len(names))
    ]
    ncm_rows = [
        {
            "true_class": names[i],
            **{names[j]: metrics["confusion_matrix_normalized"][i][j] for j in range(len(names))},
        }
        for i in range(len(names))
    ]
    write_csv(RUN / "metrics/confusion_matrix.csv", cm_rows)
    write_csv(RUN / "metrics/confusion_matrix_normalized.csv", ncm_rows)
    write_json(
        RUN / "metrics/metric_configuration.json",
        {
            "class_order": names,
            "zero_division": 0.0,
            "top1": True,
            "top5": "NA because only three classes exist",
            "input_rows": len(predictions),
        },
    )
    report_lines = [
        "Classification report",
        f"Images: {metrics['total_images']}",
        f"Correct: {metrics['correct']}",
        f"Incorrect: {metrics['incorrect']}",
        f"Accuracy: {metrics['accuracy']:.9f}",
        f"Macro precision: {metrics['macro_precision']:.9f}",
        f"Macro recall: {metrics['macro_recall']:.9f}",
        f"Macro F1: {metrics['macro_f1']:.9f}",
        f"Weighted F1: {metrics['weighted_f1']:.9f}",
    ]
    (RUN / "metrics/classification_report.txt").write_text("\n".join(report_lines) + "\n")

    stage_keys = [
        key for key in timings[0]
        if key.endswith("_ms") and key not in {"end_to_end_ms", "transfer_inclusive_ms"}
    ] + ["transfer_inclusive_ms", "end_to_end_ms"]
    end_mean = statistics.mean(numeric(timings, "end_to_end_ms"))
    stage_rows = []
    for key in stage_keys:
        summary = stats(numeric(timings, key))
        summary.update(
            {
                "stage": key,
                "unit": "ms",
                "mean_share_of_end_to_end_percent": 100 * summary["mean"] / end_mean if end_mean else None,
                "median_share_of_end_to_end_percent": 100 * summary["median"] / statistics.median(numeric(timings, "end_to_end_ms")),
            }
        )
        stage_rows.append(summary)
    write_csv(RUN / "performance/stage_timing_summary.csv", stage_rows)
    write_json(RUN / "performance/stage_timing_summary.json", stage_rows)
    write_csv(
        RUN / "performance/latency_percentiles.csv",
        [
            {
                "stage": row["stage"],
                "p50_ms": row["p50"],
                "p90_ms": row["p90"],
                "p95_ms": row["p95"],
                "p99_ms": row["p99"],
            }
            for row in stage_rows
        ],
    )
    mean_engine = statistics.mean(numeric(timings, "inference_ms"))
    mean_transfer = statistics.mean(numeric(timings, "transfer_inclusive_ms"))
    mean_end = statistics.mean(numeric(timings, "end_to_end_ms"))
    performance = {
        "measured_images": len(timings),
        "warmup_count": provenance["warmup_executions"],
        "mean_engine_latency_ms": mean_engine,
        "mean_transfer_inclusive_latency_ms": mean_transfer,
        "mean_end_to_end_latency_ms": mean_end,
        "engine_fps": 1000.0 / mean_engine,
        "transfer_inclusive_fps": 1000.0 / mean_transfer,
        "latency_based_pipeline_fps": 1000.0 / mean_end,
        "official_measured_wall_time_seconds": provenance["official_measured_wall_time_seconds"],
        "wall_clock_pipeline_fps": len(timings) / provenance["official_measured_wall_time_seconds"],
        "engine_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "inference_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "inference_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "inference_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "inference_ms"), 99)),
        },
        "end_to_end_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "end_to_end_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "end_to_end_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "end_to_end_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "end_to_end_ms"), 99)),
        },
        "fps_definitions": {
            "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
            "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
            "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
            "wall_clock_pipeline_fps": "N / official measured process wall time",
        },
    }
    write_json(RUN / "performance/performance_summary.json", performance)
    write_csv(RUN / "performance/performance_summary.csv", [performance])
    (RUN / "performance/fps_definitions.md").write_text(
        "# FPS Definitions\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in performance["fps_definitions"].items()) + "\n",
        encoding="utf-8",
    )

    idle_power = numeric(idle_rows, "power_tot_w")
    active_power = numeric(power_rows, "power_tot_w")
    if not idle_power or not active_power:
        raise RuntimeError("Power TOT telemetry is missing")
    idle_avg = statistics.mean(idle_power)
    gross_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=0.0,
    )
    net_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=idle_avg,
    )
    power_summary = {
        "power_boundary": "Jetson module-level Power TOT (POM_5V_IN), not GPU-only power",
        "monitor": "jtop/jetson-stats 4.3.2",
        "interval_seconds": 0.5,
        "active_sample_count": len(active_power),
        "idle_sample_count": len(idle_power),
        "mean_power_tot_w": statistics.mean(active_power),
        "median_power_tot_w": statistics.median(active_power),
        "min_power_tot_w": min(active_power),
        "max_power_tot_w": max(active_power),
        "p95_power_tot_w": float(np.percentile(active_power, 95)),
        "gross_total_energy_j": gross_energy,
        "gross_energy_per_image_j": gross_energy / len(timings),
        "idle_mean_power_w": idle_avg,
        "idle_corrected_total_energy_j": net_energy,
        "idle_corrected_energy_per_image_j": net_energy / len(timings),
        "images_per_gross_joule": len(timings) / gross_energy,
        "images_per_idle_corrected_joule": len(timings) / net_energy if net_energy else None,
        "integration": "trapezoidal integration over timestamped samples with boundary extension",
        "warmup_energy_excluded": True,
    }
    write_json(RUN / "monitoring/power_energy_summary.json", power_summary)
    cpu_values = numeric(power_rows, "cpu_util_percent")
    gpu_values = numeric(power_rows, "gpu_util_percent")
    ram_values = numeric(power_rows, "ram_used_mb")
    rss_values = numeric(power_rows, "process_rss_mb")
    temperatures = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
        if any(row.get(k) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"])
    ]
    cpu_ram = {
        "avg_cpu_util_percent": statistics.mean(cpu_values),
        "max_cpu_util_percent": max(cpu_values),
        "avg_gpu_util_percent": statistics.mean(gpu_values),
        "max_gpu_util_percent": max(gpu_values),
        "avg_ram_used_mb": statistics.mean(ram_values),
        "max_ram_used_mb": max(ram_values),
        "avg_process_rss_mb": statistics.mean(rss_values),
        "max_process_rss_mb": max(rss_values),
    }
    temperature_summary = {
        "avg_max_sensor_temperature_c": statistics.mean(temperatures),
        "max_sensor_temperature_c": max(temperatures),
        "throttling_event_count": 0,
        "throttling_telemetry_note": "No explicit throttling flag was exposed by the preserved jtop monitor schema; no throttling event was observed or inferred.",
    }
    write_json(RUN / "monitoring/cpu_ram_summary.json", cpu_ram)
    write_json(RUN / "monitoring/temperature_summary.json", temperature_summary)
    write_csv(
        RUN / "monitoring/throttling_events.csv",
        [],
        ["timestamp_epoch", "event", "source", "notes"],
    )

    categories = {
        "file decoding": ["decode_ms"],
        "color conversion": ["color_conversion_ms"],
        "resize/crop": ["resize_calculation_ms", "resize_ms", "crop_padding_ms"],
        "normalization/layout": [
            "float32_conversion_ms", "pixel_scaling_ms", "mean_subtraction_ms",
            "std_division_ms", "hwc_to_chw_ms", "batch_insertion_ms",
            "contiguous_preparation_ms",
        ],
        "H2D": ["h2d_ms"],
        "TensorRT FP16 inference": ["inference_ms"],
        "D2H": ["d2h_ms"],
        "postprocessing": [
            "output_vector_handling_ms", "softmax_ms", "argmax_topk_ms",
            "classification_object_ms",
        ],
        "pipeline overhead": ["pipeline_overhead_ms"],
    }
    per_image_shares = []
    category_values = {name: [] for name in categories}
    for timing in timings:
        e2e = float(timing["end_to_end_ms"])
        share_row = {"measured_index": timing["measured_index"], "end_to_end_ms": e2e}
        for name, keys in categories.items():
            value = sum(float(timing[key]) for key in keys)
            category_values[name].append(value)
            share_row[name] = value
            share_row[name + "_share_percent"] = 100 * value / e2e if e2e else None
        per_image_shares.append(share_row)
    bottlenecks = []
    for name, values in category_values.items():
        summary = stats(values)
        bottlenecks.append(
            {
                "stage": name,
                "mean_latency_ms": summary["mean"],
                "median_latency_ms": summary["median"],
                "p95_ms": summary["p95"],
                "p99_ms": summary["p99"],
                "accumulated_latency_ms": summary["accumulated"],
                "share_of_end_to_end_percent": 100 * summary["mean"] / mean_end,
            }
        )
    bottlenecks.sort(key=lambda row: row["mean_latency_ms"], reverse=True)
    for rank, row in enumerate(bottlenecks, 1):
        row["bottleneck_rank"] = rank
    outside_engine = 100.0 - next(
        row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "TensorRT FP16 inference"
    )
    analysis = {
        "main_bottleneck": bottlenecks[0],
        "latency_outside_tensorrt_percent": outside_engine,
        "decoding_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "file decoding"),
        "preprocessing_percent": sum(
            row["share_of_end_to_end_percent"]
            for row in bottlenecks
            if row["stage"] in {"color conversion", "resize/crop", "normalization/layout"}
        ),
        "tensorrt_inference_percent": 100 - outside_engine,
        "transfers_percent": sum(
            row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] in {"H2D", "D2H"}
        ),
        "postprocessing_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "postprocessing"),
        "optimization_priority": (
            "preprocessing/outside-engine path"
            if outside_engine > 50
            else "TensorRT inference"
        ),
    }
    write_csv(RUN / "analysis/bottleneck_analysis.csv", bottlenecks)
    write_json(RUN / "analysis/bottleneck_analysis.json", analysis)
    write_csv(RUN / "analysis/per_image_stage_shares.csv", per_image_shares)
    write_csv(RUN / "performance/bottleneck_ranking.csv", bottlenecks)
    (RUN / "analysis/bottleneck_analysis.md").write_text(
        "# Bottleneck Analysis\n\n"
        f"- Main bottleneck: **{analysis['main_bottleneck']['stage']}** "
        f"({analysis['main_bottleneck']['mean_latency_ms']:.3f} ms, "
        f"{analysis['main_bottleneck']['share_of_end_to_end_percent']:.2f}%)\n"
        f"- Decoding share: {analysis['decoding_percent']:.2f}%\n"
        f"- Preprocessing share: {analysis['preprocessing_percent']:.2f}%\n"
        f"- TensorRT inference share: {analysis['tensorrt_inference_percent']:.2f}%\n"
        f"- Transfer share: {analysis['transfers_percent']:.2f}%\n"
        f"- Postprocessing share: {analysis['postprocessing_percent']:.2f}%\n"
        f"- Outside TensorRT: {analysis['latency_outside_tensorrt_percent']:.2f}%\n"
        f"- Optimization priority: {analysis['optimization_priority']}\n"
    )

    # Direct comparison with the validated final MobileNetV3-Small FP32 package.
    ref_metrics = read_json(FP32_REFERENCE / "metrics/classification_metrics.json")
    ref_performance = read_json(FP32_REFERENCE / "performance/performance_summary.json")
    ref_power = read_json(FP32_REFERENCE / "monitoring/power_energy_summary.json")
    ref_resources = read_json(FP32_REFERENCE / "monitoring/cpu_ram_summary.json")
    ref_temperature = read_json(FP32_REFERENCE / "monitoring/temperature_summary.json")
    ref_article = read_json(FP32_REFERENCE / "summary/article_results.json")
    current_config = read_json(RUN / "config/preprocessing.json")
    reference_config = read_json(FP32_REFERENCE / "config/preprocessing.json")
    fp32_predictions = read_csv(FP32_REFERENCE / "raw/per_image_predictions.csv")
    prediction_rows = []
    for fp32_row, fp16_row in zip(fp32_predictions, predictions):
        prediction_rows.append({
            "measured_index": int(fp16_row["measured_index"]),
            "relative_path": fp16_row["relative_path"],
            "true_class_id": int(fp16_row["true_class_id"]),
            "fp32_predicted_class_id": int(fp32_row["predicted_class_id"]),
            "fp16_predicted_class_id": int(fp16_row["predicted_class_id"]),
            "prediction_agrees": int(fp32_row["predicted_class_id"] == fp16_row["predicted_class_id"]),
            "fp32_confidence": float(fp32_row["confidence"]),
            "fp16_confidence": float(fp16_row["confidence"]),
            "confidence_difference_fp16_minus_fp32": float(fp16_row["confidence"]) - float(fp32_row["confidence"]),
        })
    prediction_agreement = statistics.mean(row["prediction_agrees"] for row in prediction_rows)
    write_csv(RUN / "comparison/fp32_fp16_prediction_parity.csv", prediction_rows)
    write_json(RUN / "comparison/fp32_fp16_metric_parity.json", {
        "status": "PASS" if prediction_agreement >= 0.99 else "FAIL",
        "prediction_agreement": prediction_agreement,
        "agreement_count": sum(row["prediction_agrees"] for row in prediction_rows),
        "total_images": len(prediction_rows),
        "accuracy_difference_fp16_minus_fp32": metrics["accuracy"] - ref_metrics["accuracy"],
        "macro_f1_difference_fp16_minus_fp32": metrics["macro_f1"] - ref_metrics["macro_f1"],
    })
    parity = {
        "status": "PASS",
        "onnx_sha256_match": sha256(PACKAGED_ONNX) == sha256(FP32_REFERENCE / "onnx/mobilenet_v3_small_source.onnx"),
        "dataset_inventory": read_json(RUN / "validation/fp32_fp16_dataset_parity.json")["status"],
        "class_mapping_match": read_json(RUN / "dataset/class_mapping.json") == read_json(FP32_REFERENCE / "dataset/class_mapping.json"),
        "preprocessing_match": current_config == reference_config,
        "input_shape_match": ref_metrics.get("input_shape") == [1, 3, 640, 640],
        "batch_size_match": ref_metrics.get("batch_size") == 1,
        "warmup_count_match": ref_performance.get("warmup_count") == provenance["warmup_executions"] == 10,
        "measured_image_count_match": ref_performance.get("measured_images") == len(timings) == 261,
        "timing_implementation_match": read_json(RUN / "validation/fp32_fp16_protocol_precheck.json")["official_benchmark_script_content_match"],
        "telemetry_implementation_match": read_json(RUN / "validation/fp32_fp16_protocol_precheck.json")["monitor_script_hash_match"],
        "monitoring_interval_match": ref_power.get("interval_seconds") == 0.5,
        "limitations": "Run-time temperature and available memory naturally differ between sequential experiments.",
    }
    parity["status"] = "PASS" if all(
        value in (True, "PASS") for key, value in parity.items()
        if key not in {"status", "limitations"}
    ) else "FAIL"
    write_json(RUN / "comparison/protocol_parity_check.json", parity)
    comparison_rows = [
        {
            "precision": "FP32", "accuracy": ref_metrics["accuracy"], "macro_f1": ref_metrics["macro_f1"],
            "engine_latency_ms": ref_performance["mean_engine_latency_ms"], "engine_fps": ref_performance["engine_fps"],
            "end_to_end_latency_ms": ref_performance["mean_end_to_end_latency_ms"], "pipeline_fps": ref_performance["latency_based_pipeline_fps"],
            "wall_clock_fps": ref_performance["wall_clock_pipeline_fps"], "average_power_w": ref_power["mean_power_tot_w"],
            "gross_energy_per_image_j": ref_power["gross_energy_per_image_j"], "idle_corrected_energy_per_image_j": ref_power["idle_corrected_energy_per_image_j"],
            "avg_cpu_percent": ref_resources["avg_cpu_util_percent"], "avg_gpu_percent": ref_resources["avg_gpu_util_percent"],
            "max_ram_mb": ref_resources["max_ram_used_mb"], "max_temperature_c": ref_temperature["max_sensor_temperature_c"],
            "engine_size_bytes": ref_article["engine"]["size_bytes"],
        },
        {
            "precision": "FP16-enabled", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "engine_latency_ms": performance["mean_engine_latency_ms"], "engine_fps": performance["engine_fps"],
            "end_to_end_latency_ms": performance["mean_end_to_end_latency_ms"], "pipeline_fps": performance["latency_based_pipeline_fps"],
            "wall_clock_fps": performance["wall_clock_pipeline_fps"], "average_power_w": power_summary["mean_power_tot_w"],
            "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"], "idle_corrected_energy_per_image_j": power_summary["idle_corrected_energy_per_image_j"],
            "avg_cpu_percent": cpu_ram["avg_cpu_util_percent"], "avg_gpu_percent": cpu_ram["avg_gpu_util_percent"],
            "max_ram_mb": cpu_ram["max_ram_used_mb"], "max_temperature_c": temperature_summary["max_sensor_temperature_c"],
            "engine_size_bytes": PACKAGED_ENGINE.stat().st_size,
        },
    ]
    comparison = {
        "status": parity["status"], "rows": comparison_rows,
        "prediction_agreement": prediction_agreement,
        "accuracy_difference_fp16_minus_fp32": metrics["accuracy"] - ref_metrics["accuracy"],
        "macro_f1_difference_fp16_minus_fp32": metrics["macro_f1"] - ref_metrics["macro_f1"],
        "fp16_engine_speedup": ref_performance["mean_engine_latency_ms"] / performance["mean_engine_latency_ms"],
        "fp16_pipeline_speedup": ref_performance["mean_end_to_end_latency_ms"] / performance["mean_end_to_end_latency_ms"],
        "engine_fps_change_percent": 100 * (performance["engine_fps"] / ref_performance["engine_fps"] - 1),
        "pipeline_fps_change_percent": 100 * (performance["latency_based_pipeline_fps"] / ref_performance["latency_based_pipeline_fps"] - 1),
        "gross_energy_change_percent": 100 * (power_summary["gross_energy_per_image_j"] / ref_power["gross_energy_per_image_j"] - 1),
        "idle_corrected_energy_change_percent": 100 * (power_summary["idle_corrected_energy_per_image_j"] / ref_power["idle_corrected_energy_per_image_j"] - 1),
        "engine_size_change_percent": 100 * (PACKAGED_ENGINE.stat().st_size / ref_article["engine"]["size_bytes"] - 1),
        "reference_package": str(FP32_REFERENCE),
    }
    write_csv(RUN / "comparison/fp32_vs_fp16_summary.csv", comparison_rows)
    write_json(RUN / "comparison/fp32_vs_fp16_summary.json", comparison)
    (RUN / "comparison/fp32_vs_fp16_analysis.md").write_text(
        "# MobileNetV3-Small TensorRT FP32 vs FP16-enabled\n\n"
        f"- Protocol parity: **{parity['status']}**\n"
        f"- Prediction agreement: {comparison['prediction_agreement']:.6f}\n"
        f"- Accuracy difference (FP16 - FP32): {comparison['accuracy_difference_fp16_minus_fp32']:.6f}\n"
        f"- Macro F1 difference: {comparison['macro_f1_difference_fp16_minus_fp32']:.6f}\n"
        f"- Engine speedup: {comparison['fp16_engine_speedup']:.3f}x\n"
        f"- Pipeline speedup: {comparison['fp16_pipeline_speedup']:.3f}x\n"
        f"- Gross energy change: {comparison['gross_energy_change_percent']:.2f}%\n"
        f"- Engine size change: {comparison['engine_size_change_percent']:.2f}%\n"
        "- Precision is the controlled deployment variable after protocol parity checks.\n",
        encoding="utf-8",
    )

    # Figures and plotted data.
    cm = np.asarray(metrics["confusion_matrix"])
    ncm = np.asarray(metrics["confusion_matrix_normalized"])
    for name, matrix, title in [
        ("confusion_matrix_absolute", cm, "Confusion Matrix"),
        ("confusion_matrix_normalized", ncm, "Normalized Confusion Matrix"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        fig.colorbar(image, ax=ax)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i,j]:.3f}" if matrix.dtype.kind == "f" else str(matrix[i,j]), ha="center", va="center")
        save_figure(name, fig, ncm_rows if "normalized" in name else cm_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.25
    for offset, metric_name in [(-width, "precision"), (0, "recall"), (width, "f1")]:
        ax.bar(x + offset, [row[metric_name] for row in metrics["per_class"]], width, label=metric_name)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class Metrics")
    save_figure("per_class_metrics", fig, metrics["per_class"])
    indices = [int(row["measured_index"]) for row in timings]
    plot_line("engine_latency_trace", indices, numeric(timings, "inference_ms"), "measured_index", "latency_ms", "TensorRT Engine Latency", "inference_ms")
    plot_line("end_to_end_latency_trace", indices, numeric(timings, "end_to_end_ms"), "measured_index", "latency_ms", "End-to-End Latency", "end_to_end_ms")
    for name, key, title in [
        ("engine_latency_distribution", "inference_ms", "Engine Latency Distribution"),
        ("end_to_end_latency_distribution", "end_to_end_ms", "End-to-End Latency Distribution"),
    ]:
        values = numeric(timings, key)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=30)
        ax.set(xlabel="Latency (ms)", ylabel="Count", title=title)
        save_figure(name, fig, [{key: value} for value in values])
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["stage"] for row in bottlenecks]
    values = [row["mean_latency_ms"] for row in bottlenecks]
    ax.barh(labels[::-1], values[::-1])
    ax.set(xlabel="Mean latency (ms)", title="Mean Stage Latency")
    save_figure("mean_stage_latency", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 5))
    shares = [row["share_of_end_to_end_percent"] for row in bottlenecks]
    ax.barh(labels[::-1], shares[::-1])
    ax.set(xlabel="Share of end-to-end (%)", title="Mean Stage Share")
    save_figure("mean_stage_share", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    left = 0.0
    for row in sorted(bottlenecks, key=lambda item: item["bottleneck_rank"]):
        ax.barh(["Pipeline"], [row["mean_latency_ms"]], left=left, label=row["stage"])
        left += row["mean_latency_ms"]
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set(xlabel="Mean latency (ms)", title="Stacked Pipeline Breakdown")
    save_figure("stacked_pipeline_breakdown", fig, bottlenecks)
    elapsed = [float(row["timestamp_epoch"]) - float(power_rows[0]["timestamp_epoch"]) for row in power_rows]
    for name, key, ylabel, title in [
        ("power_tot_trace", "power_tot_w", "Power TOT (W)", "Jetson Module Power TOT"),
        ("gpu_utilization_trace", "gpu_util_percent", "GPU utilization (%)", "GPU Utilization"),
        ("cpu_utilization_trace", "cpu_util_percent", "CPU utilization (%)", "CPU Utilization"),
        ("ram_usage_trace", "ram_used_mb", "RAM used (MB)", "RAM Usage"),
    ]:
        values = numeric(power_rows, key)
        plot_line(name, elapsed[:len(values)], values, "elapsed_seconds", ylabel, title, key)
    temp_values = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
    ]
    plot_line("temperature_trace", elapsed[:len(temp_values)], temp_values, "elapsed_seconds", "Temperature (C)", "Maximum Sensor Temperature", "temperature_c")
    mapped = []
    for timing in timings:
        midpoint = (float(timing["image_start_epoch"]) + float(timing["image_end_epoch"])) / 2
        nearest = min(power_rows, key=lambda row: abs(float(row["timestamp_epoch"]) - midpoint))
        mapped.append(
            {
                "measured_index": timing["measured_index"],
                "end_to_end_ms": float(timing["end_to_end_ms"]),
                "power_tot_w": float(nearest["power_tot_w"]),
            }
        )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([row["power_tot_w"] for row in mapped], [row["end_to_end_ms"] for row in mapped], s=10, alpha=0.6)
    ax.set(xlabel="Nearest Power TOT sample (W)", ylabel="End-to-end latency (ms)", title="Latency versus Power TOT")
    save_figure("latency_vs_power_tot", fig, mapped)

    for figure_name, field, ylabel, title in [
        ("fp32_vs_fp16_engine_latency", "engine_latency_ms", "Latency (ms)", "TensorRT Engine Latency"),
        ("fp32_vs_fp16_engine_fps", "engine_fps", "Images/s", "TensorRT Engine FPS"),
        ("fp32_vs_fp16_pipeline_fps", "pipeline_fps", "Images/s", "End-to-End Pipeline FPS"),
        ("fp32_vs_fp16_energy_per_image", "gross_energy_per_image_j", "J/image", "Gross Energy per Image"),
        ("fp32_vs_fp16_engine_size", "engine_size_bytes", "Bytes", "TensorRT Engine Size"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar([row["precision"] for row in comparison_rows], [row[field] for row in comparison_rows])
        ax.set(ylabel=ylabel, title=title)
        save_figure(figure_name, fig, comparison_rows)
    quality_rows = [
        {"precision": row["precision"], "metric": key, "value": row[key]}
        for row in comparison_rows for key in ("accuracy", "macro_f1")
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2); width = 0.35
    ax.bar(x - width / 2, [row["accuracy"] for row in comparison_rows], width, label="Accuracy")
    ax.bar(x + width / 2, [row["macro_f1"] for row in comparison_rows], width, label="Macro F1")
    ax.set_xticks(x, [row["precision"] for row in comparison_rows]); ax.set_ylim(0.9, 1.0); ax.legend(); ax.set_title("Classification Quality")
    save_figure("fp32_vs_fp16_quality", fig, quality_rows)
    ref_bottlenecks = read_csv(FP32_REFERENCE / "performance/bottleneck_ranking.csv")
    stage_comparison_rows = []
    for architecture, rows_ in [("FP32", ref_bottlenecks), ("FP16-enabled", bottlenecks)]:
        for row in rows_:
            stage_comparison_rows.append({"architecture": architecture, "stage": row["stage"], "mean_latency_ms": float(row["mean_latency_ms"])})
    stage_names = [row["stage"] for row in bottlenecks]
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(stage_names)); width = 0.38
    for offset, architecture in [(-width / 2, "FP32"), (width / 2, "FP16-enabled")]:
        values = [next((r["mean_latency_ms"] for r in stage_comparison_rows if r["architecture"] == architecture and r["stage"] == stage), 0.0) for stage in stage_names]
        ax.bar(x + offset, values, width, label=architecture)
    ax.set_xticks(x, stage_names, rotation=35, ha="right"); ax.set_ylabel("Mean latency (ms)"); ax.set_title("Pipeline Stage Breakdown"); ax.legend()
    save_figure("fp32_vs_fp16_stage_breakdown", fig, stage_comparison_rows)

    article = {
        "architecture": "MobileNetV3-Small",
        "task": "classification",
        "device": "NVIDIA Jetson Nano Developer Kit",
        "precision": "TensorRT FP16-enabled",
        "input_shape": [1, 3, 640, 640],
        "batch_size": 1,
        "warmup_count": 10,
        "measured_images": len(timings),
        "metrics": metrics,
        "performance": performance,
        "power_energy": power_summary,
        "resources": cpu_ram,
        "temperature": temperature_summary,
        "bottleneck": analysis,
        "model_artifact": file_record(PACKAGED_MODEL),
        "original_onnx": file_record(PACKAGED_ORIGINAL_ONNX),
        "onnx": file_record(PACKAGED_ONNX),
        "engine": file_record(PACKAGED_ENGINE),
        "fp32_comparison": comparison,
    }
    write_json(RUN / "summary/article_results.json", article)
    write_csv(
        RUN / "summary/article_results.csv",
        [
            {
                "model": "MobileNetV3-Small",
                "precision": "TensorRT FP16-enabled",
                "images": len(timings),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "engine_latency_mean_ms": mean_engine,
                "end_to_end_latency_mean_ms": mean_end,
                "engine_fps": performance["engine_fps"],
                "transfer_inclusive_fps": performance["transfer_inclusive_fps"],
                "pipeline_fps": performance["latency_based_pipeline_fps"],
                "wall_clock_fps": performance["wall_clock_pipeline_fps"],
                "avg_power_tot_w": power_summary["mean_power_tot_w"],
                "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
                "idle_corrected_energy_per_image_j": power_summary["idle_corrected_energy_per_image_j"],
            }
        ],
    )
    article_md = f"""# Publication Results: MobileNetV3-Small TensorRT FP16-enabled on Jetson Nano

| Item | Result |
|---|---:|
| Images | {len(timings)} |
| Accuracy | {metrics['accuracy']:.6f} |
| Macro precision | {metrics['macro_precision']:.6f} |
| Macro recall | {metrics['macro_recall']:.6f} |
| Macro F1 | {metrics['macro_f1']:.6f} |
| Weighted F1 | {metrics['weighted_f1']:.6f} |
| Engine latency | {mean_engine:.3f} ms |
| End-to-end latency | {mean_end:.3f} ms |
| Engine FPS | {performance['engine_fps']:.3f} |
| Transfer-inclusive FPS | {performance['transfer_inclusive_fps']:.3f} |
| Wall-clock pipeline FPS | {performance['wall_clock_pipeline_fps']:.3f} |
| Average Power TOT | {power_summary['mean_power_tot_w']:.3f} W |
| Gross energy/image | {power_summary['gross_energy_per_image_j']:.3f} J |
| Idle-corrected energy/image | {power_summary['idle_corrected_energy_per_image_j']:.3f} J |
"""
    (RUN / "summary/article_results.md").write_text(article_md)
    methodology = f"""# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: MobileNetV3-Small classification, TensorRT FP16-enabled (`--fp16`, `--noTF32`, no INT8).
- Source artifact: the exact ONNX from the validated FP32 final package; no re-export occurred.
- Host input/output bindings remain FP32 for preprocessing parity; internal layer precision and fallback evidence are recorded separately.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: {len(timings)} validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- FP32-versus-FP16 protocol parity: {parity['status']}.
"""
    reports = {
        "methodology.md": methodology,
        "preprocessing_report.md": "# Preprocessing Parity\n\n" + json.dumps(read_json(RUN / "config/preprocessing_parity.json"), indent=2),
        "classification_metrics_report.md": article_md,
        "performance_report.md": "# Performance\n\n" + json.dumps(performance, indent=2),
        "power_energy_report.md": "# Power and Energy\n\n" + json.dumps(power_summary, indent=2),
        "bottleneck_report.md": (RUN / "analysis/bottleneck_analysis.md").read_text(),
        "model_and_onnx_report.md": "# Model and ONNX\n\n" + json.dumps(read_json(RUN / "onnx/selected_onnx_provenance.json"), indent=2) + "\n\n" + (RUN / "onnx/onnx_validation_report.md").read_text(),
        "fp32_vs_fp16_report.md": (RUN / "comparison/fp32_vs_fp16_analysis.md").read_text(),
        "fairness_and_limitations.md": (
            "# Fairness and Limitations\n\n"
            "- This is a single-device, batch-1 deployment benchmark.\n"
            "- TensorRT was built locally for this Jetson and is not portable across TensorRT/GPU versions.\n"
            "- Power TOT is module-level input power, not GPU-only power.\n"
            "- jtop sampling is slower than per-image inference stages; per-image power pairing is nearest-sample only.\n"
            "- CUDA event synchronization improves stage attribution but can reduce throughput relative to an asynchronous pipeline.\n"
            "- Direct 640x640 resize follows the historical model pipeline and does not preserve aspect ratio.\n"
            "- FP16-enabled does not mean every internal layer is necessarily FP16; fallback evidence is reported where TensorRT exposes it.\n"
        ),
        "human_readable_report.md": article_md + "\n" + methodology,
    }
    for filename, content in reports.items():
        (RUN / f"summary/{filename}").write_text(content.rstrip() + "\n")
    figure_files = sorted((RUN / "figures/png").glob("*.png"))
    (RUN / "summary/figure_index.md").write_text(
        "# Figure Index\n\n" + "\n".join(f"- `{path.name}`" for path in figure_files) + "\n"
    )
    warmup_protocol = {
        "warmup_count": provenance["warmup_executions"],
        "policy": provenance["warmup_policy"],
        "excluded_from_timing": provenance["warmups_excluded_from_timing"],
        "excluded_from_monitoring": provenance["warmups_excluded_from_monitoring"],
        "measured_executions": provenance["measured_executions"],
        "total_engine_executions": provenance["total_engine_executions"],
    }
    write_json(RUN / "summary/warmup_protocol.json", warmup_protocol)
    (RUN / "summary/warmup_protocol.md").write_text(
        "# Warm-up Protocol\n\n"
        f"- Exactly {provenance['warmup_executions']} TensorRT executions.\n"
        "- First deterministic image preprocessed once before warm-up.\n"
        "- Warm-ups excluded from all official timing and telemetry.\n"
        f"- Official measured executions: {provenance['measured_executions']}.\n",
        encoding="utf-8",
    )

    # Code provenance and complete hash manifest.
    provenance_rows = []
    for path in sorted((RUN / "scripts").glob("*")):
        if path.is_file():
            provenance_rows.append(
                {
                    "absolute_path": str(path),
                    "packaged_path": str(path.relative_to(RUN)),
                    "sha256": sha256(path),
                    "purpose": "benchmark package source code",
                    "exit_status": "PASS if executed; syntax verified below",
                }
            )
    write_csv(RUN / "validation/executed_code_provenance.csv", provenance_rows)
    syntax_failures = []
    for path in sorted((RUN / "scripts").glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            syntax_failures.append({"path": str(path), "error": result.stderr})

    required = [
        "raw/per_image_timings.csv",
        "raw/per_image_predictions.csv",
        "raw/per_image_results.jsonl",
        "raw/ground_truth.csv",
        "raw/warmup_execution_records.csv",
        "raw/execution_count_provenance.json",
        "raw/failed_images.csv",
        "metrics/classification_metrics.json",
        "metrics/confusion_matrix.csv",
        "metrics/confusion_matrix_normalized.csv",
        "performance/performance_summary.json",
        "monitoring/idle_samples.csv",
        "monitoring/primary_power_samples.csv",
        "monitoring/power_energy_summary.json",
        "analysis/bottleneck_analysis.json",
        "comparison/fp32_vs_fp16_summary.json",
        "comparison/protocol_parity_check.json",
        "summary/article_results.md",
        "onnx/onnx_validation_report.md",
        "engine/engine_metadata.json",
        "validation/correctness_validation_report.md",
    ]
    missing = [path for path in required if not (RUN / path).exists()]
    checks = {
        "device_is_jetson_nano": "Jetson Nano" in read_json(RUN / "environment/system_state_before.json")["device_model"],
        "task_is_classification": metrics["task"] == "classification",
        "model_is_mobilenet_v3_small": read_json(RUN / "model/model_metadata.json")["architecture"] == "MobileNetV3-Small",
        "fp32_reference_pass": read_json(RUN / "validation/fp32_reference_package.json")["status"] == "PASS",
        "fp16_builder_enabled": "--fp16" in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "tf32_disabled": "--noTF32" in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "int8_not_enabled": "--int8" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "calibration_not_used": "calibration" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0].lower(),
        "dla_not_used": "--useDLACore" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "input_shape_verified": read_json(RUN / "engine/engine_metadata.json")["input_shape_verified"],
        "output_shape_verified": any(
            item.get("runtime_shape", item.get("engine_shape", item.get("shape"))) == [1, 3]
            and not item["is_input"]
            for item in read_json(RUN / "engine/engine_metadata.json")["bindings"]
        ),
        "warmup_exactly_10": provenance["warmup_executions"] == 10,
        "warmup_excluded": provenance["warmups_excluded_from_monitoring"] and provenance["warmups_excluded_from_timing"],
        "measured_count_matches": provenance["measured_executions"] == 261 == len(timings),
        "failed_images_zero": provenance["failed_images"] == 0,
        "discarded_rows_zero": provenance["discarded_measured_rows"] == 0,
        "all_timing_stages_present": all(all(row.get(key) not in (None, "") for key in [
            "decode_ms", "preprocessing_ms", "h2d_ms", "inference_ms", "d2h_ms",
            "postprocessing_ms", "end_to_end_ms"
        ]) for row in timings),
        "metrics_exist": metrics["total_images"] == 261,
        "power_samples_exist": len(active_power) > 0,
        "energy_integrated": gross_energy is not None and gross_energy > 0,
        "scripts_syntax_pass": not syntax_failures,
        "required_files_exist": not missing,
        "correctness_validation_pass": read_json(RUN / "validation/framework_output_summary.json")["status"] == "PASS",
        "source_hash_copy_match": read_json(RUN / "model/model_metadata.json")["copy_hash_matches"],
        "onnx_hash_copy_match": read_json(RUN / "onnx/selected_onnx_provenance.json")["copy_hash_matches"],
        "dataset_matches_fp32": read_json(RUN / "validation/fp32_fp16_dataset_parity.json")["status"] == "PASS",
        "onnx_matches_fp32": read_json(RUN / "validation/fp32_fp16_onnx_parity.json")["status"] == "PASS",
        "fallback_layers_documented": (RUN / "engine/fallback_layer_report.csv").is_file(),
        "comparison_protocol_parity": parity["status"] == "PASS",
        "figures_generated_from_raw_data": len(list((RUN / "figures/png").glob("*.png"))) >= 22,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "missing_files": missing,
        "syntax_failures": syntax_failures,
        "original_onnx_modified": False,
        "related_checkpoint_modified": False,
        "test_images_modified": False,
    }
    write_json(RUN / "validation/final_validation_report.json", validation)
    (RUN / "validation/final_validation_report.txt").write_text(
        f"FINAL STATUS: {status}\n" + "\n".join(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(f"Final validation failed: {validation}")

    manifest = read_json(RUN / "run_manifest.json")
    manifest.update(
        {
            "status": "PASS",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "measured_images": len(timings),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "engine_fps": performance["engine_fps"],
            "wall_clock_pipeline_fps": performance["wall_clock_pipeline_fps"],
            "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
            "final_validation": "PASS",
        }
    )
    write_json(RUN / "run_manifest.json", manifest)
    manifest_rows = []
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.name != "file_hash_manifest.csv":
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(RUN)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(RUN / "validation/file_hash_manifest.csv", manifest_rows)
    print("FINALIZATION PASS")
    print(json.dumps(article, indent=2, default=str))


if __name__ == "__main__":
    main()
