import json
import os
import random
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy.special
from onnxruntime import InferenceSession
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from pipeline.utility import (
    generate_report,
    get_support_list,
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)
import argparse


class ModelType(Enum):
    SMALL_MODEL = 0
    BIG_MODEL = 1


class FullPipelineMonteCarloSimulation:
    def __init__(
        self,
        small_model_path: str,
        big_model_path: str,
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        global_total_support_list: Dict[str, int],
        small_model_species_labels: Dict[int, str],
        big_model_species_labels: Dict[int, str],
        small_model_input_size: Tuple[int, int] = (224, 224),
        big_model_input_size: Tuple[int, int] = (299, 299),
        is_big_inception_v3: bool = True,
        providers: List[str] = ["CPUExecutionProvider", "CPUExecutionProvider"],
        single_model_mode: bool = False,  # <-- add this
    ) -> None:
        self.small_session = ort.InferenceSession(
            small_model_path, providers=[providers[0]]
        )
        self.big_session = ort.InferenceSession(
            big_model_path, providers=[providers[1]]
        )
        self.small_input_name = self.small_session.get_inputs()[0].name
        self.big_input_name = self.big_session.get_inputs()[0].name
        self.small_input_size = small_model_input_size
        self.big_input_size = big_model_input_size

        self.global_data_manifests = global_data_manifests
        self.global_species_labels = global_species_labels
        self.global_species_names = list(self.global_species_labels.values())
        self.global_total_support_list = global_total_support_list
        self.global_labels_images: Dict[int, List[str]] = defaultdict(list)
        for image_path, species_id in self.global_data_manifests:
            self.global_labels_images[species_id].append(image_path)
        self.global_total_images = sum(
            len(imgs) for imgs in self.global_labels_images.values()
        )
        self.global_species_probs = {
            int(species_id): len(images) / self.global_total_images
            for species_id, images in self.global_labels_images.items()
        }

        self.not_belong_to_global_idx = len(self.global_species_labels)

        self.small_species_labels: Dict[int, str] = small_model_species_labels
        self.small_other_label = self._get_small_model_other_label()
        self.big_species_labels: Dict[int, str] = big_model_species_labels
        self.big_species_name = list(self.big_species_labels.values())

        self.is_big_incv3 = is_big_inception_v3
        self.single_model_mode = single_model_mode  # <-- add this

    def _get_small_model_other_label(self) -> int:
        species_labels_flip: Dict[str, int] = dict(
            (v, k) for k, v in self.small_species_labels.items()
        )
        return species_labels_flip.get("Other", -1)

    def _is_prediction_belongs_to_global_dataset(self, prediction: int) -> bool:
        species_name_big: str | None = self.big_species_labels.get(prediction, None)

        if species_name_big is None:
            print(
                f"Cannot determine species name from big model prediction: {prediction}"
            )
            return False

        if species_name_big not in self.global_species_names:
            return False

        return True

    def _translate_prediction_to_global_label(
        self, prediction: int, model_type: ModelType
    ):
        if model_type == ModelType.BIG_MODEL:
            big_species_label = self.big_species_labels.get(prediction, None)
            global_species_labels = list(
                filter(
                    lambda key: self.global_species_labels[key] == big_species_label,
                    self.global_species_labels,
                )
            )
            if not global_species_labels:
                print(
                    f"[Warning] Could not map species from big pred {big_species_labels} to global label"
                )
                return self.not_belong_to_global_idx
            return global_species_labels[0]
        else:
            small_species_label = self.small_species_labels.get(prediction, None)
            global_species_labels = list(
                filter(
                    lambda key: self.global_species_labels[key] == small_species_label,
                    self.global_species_labels,
                )
            )
            if not global_species_labels:
                print(
                    f"[Warning] Could not map species from small pred {small_species_labels} to global label"
                )
                return self.not_belong_to_global_idx
            return global_species_labels[0]

    def _create_stratified_weighted_sample(self, sample_size: int):
        sampled_species = list(self.global_species_labels.keys())
        remaining_k: int = sample_size - len(sampled_species)
        sampled_species += random.choices(
            population=sampled_species,
            weights=[
                self.global_species_probs[int(sid)]
                for sid in self.global_species_labels.keys()
            ],
            k=remaining_k,
        )
        random.shuffle(sampled_species)
        return [int(label) for label in sampled_species]

    def _infer_one(
        self, model_type: ModelType, image_path: str
    ) -> Optional[Tuple[int, float]]:
        if model_type == ModelType.BIG_MODEL:
            session: InferenceSession = self.big_session
            input_size = self.big_input_size
            input_name = self.big_input_name
            if self.is_big_incv3:
                is_incv3 = True
            else:
                is_incv3 = False
        else:
            session = self.small_session
            input_size = self.small_input_size
            input_name = self.small_input_name
            is_incv3 = False

        try:
            img = preprocess_eval_opencv(
                image_path, *input_size, is_inception_v3=is_incv3
            )
            outputs = session.run(None, {input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None

    def infer_with_routing(self, image_path: str, ground_truth: int):
        if self.single_model_mode:
            # Only use the big model (ConvNeXt)
            big_result = self._infer_one(ModelType.BIG_MODEL, image_path)
            if big_result is None:
                print(f"Big model returns no result for {image_path}")
                return None
            if not self._is_prediction_belongs_to_global_dataset(big_result[0]):
                return ground_truth, self.not_belong_to_global_idx, 1

            global_pred = self._translate_prediction_to_global_label(
                big_result[0], ModelType.BIG_MODEL
            )
            return ground_truth, global_pred, 1
        else:
            # Original two-stage logic
            small_result = self._infer_one(ModelType.SMALL_MODEL, image_path)
            if small_result is None:
                print(f"Small model returns no result for {image_path}")
                return None
            small_pred, _ = small_result

            if small_pred == self.small_other_label:
                big_result = self._infer_one(ModelType.BIG_MODEL, image_path)
                if big_result is None:
                    print(f"Big model returns no result for {image_path}")
                    return None
                if not self._is_prediction_belongs_to_global_dataset(big_result[0]):
                    return ground_truth, self.not_belong_to_global_idx, 1

                global_pred = self._translate_prediction_to_global_label(
                    big_result[0], ModelType.BIG_MODEL
                )
                return ground_truth, global_pred, 1
            else:
                global_pred = self._translate_prediction_to_global_label(
                    small_result[0], ModelType.SMALL_MODEL
                )
                return ground_truth, global_pred, 0

    def run(self, num_runs: int, sample_size: int = 1000, save_path=None):
        other_preds = 0
        all_true, all_pred = [], []

        # Timing lists for each model
        small_times = []
        big_times = []
        total_images = 0

        pipeline_start = time.perf_counter()

        for run in range(num_runs):
            y_true: List[int] = []
            y_pred: List[int] = []
            sampled_species = self._create_stratified_weighted_sample(sample_size)

            for species_id in tqdm(
                sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False
            ):
                image_list = self.global_labels_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)

                # Inference with timing
                if self.single_model_mode:
                    t0 = time.perf_counter()
                    result = self.infer_with_routing(image_path, species_id)
                    t1 = time.perf_counter()
                    big_times.append(t1 - t0)
                else:
                    t0 = time.perf_counter()
                    small_result = self._infer_one(ModelType.SMALL_MODEL, image_path)
                    t1 = time.perf_counter()
                    small_times.append(t1 - t0)
                    if small_result is None:
                        print(f"Small model returns no result for {image_path}")
                        continue
                    small_pred, _ = small_result

                    if small_pred == self.small_other_label:
                        t2 = time.perf_counter()
                        big_result = self._infer_one(ModelType.BIG_MODEL, image_path)
                        t3 = time.perf_counter()
                        big_times.append(t3 - t2)
                        if big_result is None:
                            print(f"Big model returns no result for {image_path}")
                            continue
                        result = self.infer_with_routing(image_path, species_id)
                    else:
                        result = self.infer_with_routing(image_path, species_id)

                if result is not None:
                    ground_truth, pred, is_other_small_model = result
                    other_preds += is_other_small_model
                    y_true.append(ground_truth)
                    y_pred.append(pred)
                    total_images += 1

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        pipeline_end = time.perf_counter()
        total_time = pipeline_end - pipeline_start
        avg_small_time = sum(small_times) / len(small_times) if small_times else 0
        avg_big_time = sum(big_times) / len(big_times) if big_times else 0
        throughput = total_images / total_time if total_time > 0 else 0

        total_support_list = get_support_list(
            self.global_total_support_list, self.global_species_names
        )
        num_pred_outside_global = sum(
            [1 for pred in all_pred if pred == self.not_belong_to_global_idx]
        )
        print(f"Total 'Other' prediction by small model: {other_preds}")
        print(f"Total prediction outside the test set: {num_pred_outside_global}")

        if save_path:
            accuracy = accuracy_score(all_true, all_pred)
            df = generate_report(
                all_true,
                all_pred,
                self.global_species_names,
                total_support_list,
                float(accuracy),
            )

            # Save benchmark metrics
            metrics = {
                "total_images": total_images,
                "total_time_sec": total_time,
                "throughput_fps": throughput,
                "avg_small_model_time_ms": avg_small_time * 1000,
                "avg_big_model_time_ms": avg_big_time * 1000,
                "small_model_calls": len(small_times),
                "big_model_calls": len(big_times),
                "accuracy": accuracy,
                "other_preds": other_preds,
                "pred_outside_global": num_pred_outside_global,
            }
            metrics_df = pd.DataFrame([metrics])
            os.makedirs(save_path, exist_ok=True)
            metrics_df.to_csv(
                os.path.join(save_path, "pipeline_benchmark_metrics.csv"), index=False
            )

            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                model_name = "pipeline_convnext_full"
                df.to_csv(os.path.join(save_path, f"{model_name}.csv"))

            # Confusion matrix
            cm = confusion_matrix(
                all_true, all_pred, labels=list(range(len(self.global_species_names)))
            )
            cm_df = pd.DataFrame(
                cm, index=self.global_species_names, columns=self.global_species_names
            )
            cm_df.to_csv(os.path.join(save_path, "confusion_matrix.csv"))

            print("\n=== Benchmark Results ===")
            for k, v in metrics.items():
                print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["small", "big", "both"],
        default="both",
        help="Which model(s) to benchmark: small, big, or both (two-stage pipeline)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=160,
        help="Input size for both models (square)",
    )
    parser.add_argument(
        "--runs", type=int, default=15, help="Number of Monte Carlo runs"
    )
    parser.add_argument("--samples", type=int, default=1000, help="Sample size per run")
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="benchmark",
        help="Prefix for output CSV files",
    )
    args = parser.parse_args()

    threshold = 0.5
    small_image_data, _, _, small_species_labels, _ = manifest_generator_wrapper(
        threshold
    )
    global_image_data, _, _, global_species_labels, global_species_composition = (
        manifest_generator_wrapper(1.0)
    )
    with open("./haute_garonne/dataset_species_labels.json") as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k): v for k, v in big_species_labels.items()}

    # Model selection logic
    if args.model == "big":
        single_model_mode = True
        model_desc = "convnext_only"
    elif args.model == "small":
        single_model_mode = True
        model_desc = "mcunet_only"
    else:
        single_model_mode = False
        model_desc = "mcunet_convnext_pipeline"

    # Input size logic (you can expand this if you want different sizes for each model)
    input_size = (args.input_size, args.input_size)

    pipeline = FullPipelineMonteCarloSimulation(
        "models/mcunet-in2-haute-garonne_8_best.onnx",
        "models/convnext_full_insect_best.onnx",
        global_image_data,
        global_species_labels,
        global_species_composition,
        small_species_labels,
        big_species_labels,
        is_big_inception_v3=False,
        small_model_input_size=input_size,
        big_model_input_size=input_size,
        providers=["CUDAExecutionProvider", "CUDAExecutionProvider"],
        single_model_mode=single_model_mode,
    )
    out_dir = f"./{args.out_prefix}_{model_desc}_{args.input_size}"
    pipeline.run(args.runs, args.samples, out_dir)
