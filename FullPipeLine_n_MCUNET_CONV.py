import json
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy.special
from onnxruntime import InferenceSession
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from pipeline.utility import (
    generate_report,
    get_support_list,
    manifest_generator_wrapper,
    preprocess_eval_opencv,
)
import time
import statistics as stats
import argparse


class ModelType(Enum):
    SMALL_MODEL = 0
    BIG_MODEL = 1


class NFullPipelineMonteCarloSimulation:
    """
    This is the scaled version of the FullPipelineMonteCarloSimulation.
    In short, it is the same as the FullPipelineMonteCarloSimulation but with n mcunet models and 1 convnext model.

    Args:
        small_model_path (str): Path to the small model ONNX file.
        big_model_path (str): Path to the big model ONNX file.
        global_data_manifests (List[Tuple[str, int]]): List of tuples containing image paths and their corresponding species IDs.
        global_species_labels (Dict[int, str]): Dictionary mapping species IDs to species names.
        global_total_support_list (Dict[str, int]): Dictionary mapping species names to their total support count.
        small_model_species_labels (Dict[int, str]): Dictionary mapping species IDs to species names for the small model.
        big_model_species_labels (Dict[int, str]): Dictionary mapping species IDs to species names for the big model.
        small_model_input_size (Tuple[int, int]): Input size for the small model. Default is (224, 224).
        big_model_input_size (Tuple[int, int]): Input size for the big model. Default is (299, 299).
        is_big_inception_v3 (bool): Flag indicating if the big model is Inception V3. Default is True.
        providers (List[str]): List of ONNX Runtime providers to use for inference. Default is ["CPUExecutionProvider", "CPUExecutionProvider"].

    """

    def __init__(
        self,
        small_model_path: list,
        big_model_path: str,
        number_of_small_models: int,
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        global_total_support_list: Dict[str, int],
        small_model_species_labels: Dict[int, str],
        big_model_species_labels: Dict[int, str],
        small_model_input_size: Tuple[int, int] = (224, 224),
        big_model_input_size: Tuple[int, int] = (299, 299),
        is_big_inception_v3: bool = True,
        providers: List[str] = ["CPUExecutionProvider", "CPUExecutionProvider"],
    ) -> None:
        # self.small_session = ort.InferenceSession(
        #     small_model_path, providers=[providers[0]]
        # )
        self.providers = providers
        self.small_model_path: List[str] = small_model_path
        self.big_session = ort.InferenceSession(
            big_model_path, providers=[self.providers[1]]
        )
        # self.small_input_name = self.small_session.get_inputs()[0].name
        self.number_of_small_models = number_of_small_models
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
        self.small_species_preds: Dict = {}
        self.global_species_preds: list = []

    def _get_small_model_other_label(self) -> int:
        """
        Get the label for 'Other' species in the small model.
        """
        species_labels_flip: Dict[str, int] = dict(
            (v, k) for k, v in self.small_species_labels.items()
        )
        return species_labels_flip.get("Other", -1)

    def _is_prediction_belongs_to_global_dataset(self, prediction: int) -> bool:
        """
        Check if the prediction belongs to the global dataset.
        Args:
            prediction (int): The predicted species label.
        Returns:
                bool: True if the prediction belongs to the global dataset, False otherwise.

        """
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
    ) -> int:
        """
        Map the prediction from the small or big model to the global species label.
        Args:
            prediction (int): The predicted species label from the model.
            model_type (ModelType): The type of model (small or big).
        Returns:

        int: The global species label corresponding to the prediction.
        """
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

    def _create_stratified_weighted_sample(self, sample_size: int) -> List[int]:
        """
        Create a stratified weighted sample of species labels based on their probabilities.
        Args:
            sample_size (int): The desired size of the sample.
        Returns:
                List[int]: A list of sampled species labels.
        """
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
        self, model_type: ModelType, image_path: str, model_path: Optional[str] = None
    ) -> Optional[Tuple[int, float]]:
        """
        Perform inference on a single image using the specified model type.
        Args:
            model_type (ModelType): The type of model to use for inference (small or big).
            image_path (str): The path to the image to be processed.
        Returns:
            Optional[Tuple[int, float]]: A tuple containing the predicted species label and its probability,
            or None if an error occurs during inference.
        """
        if model_type == ModelType.BIG_MODEL:
            session: InferenceSession = self.big_session
            input_size = self.big_input_size
            input_name = self.big_input_name
            if self.is_big_incv3:
                is_incv3 = True
            else:
                is_incv3 = False
        else:
            session = ort.InferenceSession(model_path, providers=[self.providers[0]])  # type: ignore
            input_size = self.small_input_size
            input_name = session.get_inputs()[0].name
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

    def infer_with_routing(
        self, image_path: str, ground_truth: int
    ) -> Optional[Tuple[int, int, int]]:
        """
        Use Routing strategy with n small models:
        - Infer with small model first, if the result is 'Other', then infer with the next small models
        - After every small models return "Other", infer with big model.

        Args:
            image_path (str): The path to the image to be processed.
            ground_truth (int): The ground truth species label for the image.
        Returns:
            Optional[Tuple[int, int, bool]]: A tuple containing the predicted species label,
            the ground truth species label, and a boolean indicating if the small model was used.
        """
        i = 0
        for model_path in self.small_model_path:
            print(
                f"Running small model {model_path}/{self.number_of_small_models} for {image_path}"
            )
            small_result = self._infer_one(
                ModelType.SMALL_MODEL, image_path, model_path
            )
            if small_result is None:
                print(f"Small model returns no result for {image_path}")
                return None
            small_pred, _ = small_result
            # Check if the small model prediction is 'Other'
            # If it is, continue to the next small model
            if small_pred == self.small_other_label:
                i += 1
                continue
            # If the small model prediction is not 'Other', return the result
            return small_result[0], ground_truth, i

        print(
            f"All small models returned 'Other' for {image_path}, switching to big model"
        )
        big_result = self._infer_one(ModelType.BIG_MODEL, image_path)
        if big_result is None:
            print(f"Big model returns no result for {image_path}")
            return None
        return big_result[0], ground_truth, i

    def run(
        self,
        num_runs: int,
        sample_size: int = 1000,
        save_path=None,
        # model_type: str = "both",
    ) -> None:
        other_preds = 0
        all_true, all_pred = [], []
        time_per_image = []
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
                start = time.perf_counter()
                result = self.infer_with_routing(image_path, species_id)
                end = time.perf_counter()
                # if result is None:
                #     print(f"Model returns no result for {image_path}")
                #     continue
                if result is not None:
                    pred, gt, small_counter = result
                    time_per_image.append(end - start)
                    if small_counter < self.number_of_small_models:
                        # Save the small model prediction and the small_counter into the small_species_preds dict
                        self.small_species_preds[small_counter] = pred

                        global_pred = self._translate_prediction_to_global_label(
                            pred, ModelType.SMALL_MODEL
                        )
                        y_true.append(gt)
                        y_pred.append(global_pred)

                    else:
                        self.global_species_preds.append(pred)
                        global_pred = self._translate_prediction_to_global_label(
                            pred, ModelType.BIG_MODEL
                        )
                        y_true.append(gt)
                        y_pred.append(global_pred)

                # Map the result with the global species labels for evaluations
                # if result is not None:
                #     ground_truth, pred, is_other_small_model = result
                #     other_preds += is_other_small_model
                #     y_true.append(ground_truth)
                #     y_pred.append(pred)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        total_support_list = get_support_list(
            self.global_total_support_list, self.global_species_names
        )
        # num_pred_outside_global = sum(
        #     [1 for pred in all_pred if pred == self.not_belong_to_global_idx]
        # )
        print(f"Total 'Other' prediction by small model: {other_preds}")
        # print(f"Total prediction outside the test set: {num_pred_outside_global}")
        print(f"Average time per image: {stats.fmean(time_per_image) * 1000:.2f} ms")
        print(f"Throughput: {1.0 / stats.fmean(time_per_image):.2f} fps")
        # total_support_list.append(num_pred_outside_global)

        if save_path:
            accuracy = accuracy_score(all_true, all_pred)
            print(f"Accuracy: {accuracy:.4f}")
            df = generate_report(
                all_true,
                all_pred,
                self.global_species_names,
                total_support_list,
                float(accuracy),
            )

            os.makedirs(save_path, exist_ok=True)

            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                model_name = "pipeline_mcunet_convnext_full"
                df.to_csv(os.path.join(save_path, f"{model_name}.csv"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run the full pipeline Monte Carlo simulation for bird and insect species classification."
    )
    arg_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for species classification. Default is 0.5.",
    )
    arg_parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for the Monte Carlo simulation. Default is 1.",
    )
    arg_parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Sample size for each run of the simulation. Default is 1000.",
    )
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="./baseline_benchmark",
        help="Path to save the results. Default is './baseline_benchmark'.",
    )
    arg_parser.add_argument(
        "--model_type",
        type=str,
        choices=["small", "big", "both"],
        default="both",
        help="Type of model to run the simulation with. Options are 'small', 'big', or 'both'. Default is 'both'.",
    )
    arg_parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider", "CPUExecutionProvider"],
        help="List of ONNX Runtime providers to use. Default is ['CUDAExecutionProvider', 'CUDAExecutionProvider'].",
    )
    arg_parser.add_argument(
        "--small_model_input_size",
        type=int,
        nargs=2,
        default=(160, 160),
        help="Input size for the small model. Default is (160, 160).",
    )
    arg_parser.add_argument(
        "--big_model_input_size",
        type=int,
        nargs=2,
        default=(160, 160),
        help="Input size for the big model. Default is (160, 160).",
    )

    args = arg_parser.parse_args()

    threshold = args.threshold

    small_image_data, _, _, small_species_labels, _ = manifest_generator_wrapper(
        threshold
    )
    global_image_data, _, _, global_species_labels, global_species_composition = (
        manifest_generator_wrapper(1.0)
    )
    with open("haute_garonne/dataset_species_labels.json") as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k): v for k, v in big_species_labels.items()}
    # Create a list of small model paths

    small_model_paths = [
        f"models/mcunet-in2_haute_garonne_0.5_{i}_best.onnx" for i in range(0, 22, 7)
    ]
    big_model_path = "models/convnext_full_insect_best.onnx"
    pipeline = NFullPipelineMonteCarloSimulation(
        small_model_path=small_model_paths,
        big_model_path=big_model_path,
        number_of_small_models=len(small_model_paths),
        global_data_manifests=global_image_data,
        global_species_labels=global_species_labels,
        global_total_support_list=global_species_composition,
        small_model_species_labels=small_species_labels,
        big_model_species_labels=big_species_labels,
        small_model_input_size=args.small_model_input_size,
        big_model_input_size=args.big_model_input_size,
        is_big_inception_v3=True if "inception" in big_model_path else False,
        providers=args.providers,
    )

    pipeline.run(1, 1000, "./baseline_benchmark")
