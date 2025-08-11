import json
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from cv2.detail import resultRoi
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
    calculate_FN,
)
import time
import statistics as stats
import argparse
from data_prep.data_loader import DatasetCreator
from torch import Tensor


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
        small_model_path: List[str],
        big_model_path: str,
        small_data_manifests: List[str],
        global_data_manifests: List[Tuple[str, int]],
        global_species_labels: Dict[int, str],
        global_total_support_list: Dict[str, int],
        small_model_species_labels: List[Dict[int, str]],
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
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 4
        self.small_data_manifests = small_data_manifests
        self.big_session = ort.InferenceSession(
            big_model_path, session_options, providers=[self.providers[1]]
        )
        # self.small_input_name = self.small_session.get_inputs()[0].name
        self.number_of_small_models = len(small_model_path)
        self.big_input_name = self.big_session.get_inputs()[0].name
        self.small_input_size = small_model_input_size
        self.big_input_size = big_model_input_size
        self.global_data_manifests = global_data_manifests
        self.global_species_labels = global_species_labels
        self.global_species_names = list(self.global_species_labels.values())
        self.global_total_support_list = global_total_support_list
        self.global_labels_images: Dict[int, List[str]] = defaultdict(list)

        # for image_path, species_id in self.global_data_manifests:
        #     # print(f"Type of sample: {type(sample)}")
        #     self.global_labels_images[species_id].append(image_path)

        self.global_total_images = sum(
            len(imgs) for imgs in self.global_labels_images.values()
        )

        self.global_species_probs = {
            species_id: len(images) / self.global_total_images
            for species_id, images in self.global_labels_images.items()
        }
        # print(f"Global species labels: {self.global_species_labels}")

        self.not_belong_to_global_idx = len(self.global_species_labels)

        self.small_species_labels: List[Dict[int, str]] = small_model_species_labels
        self.small_other_labels: List[int] = []
        for small_species_lab in self.small_species_labels:
            other_label = self._get_small_model_other_label(small_species_lab)
            if other_label != -1:
                self.small_other_labels.append(other_label)
            else:
                print(
                    f"Warning: 'Other' label not found in small model species labels: {small_species_lab}"
                )
        # print(f"Small other labels: {self.small_other_labels}")
        # print(f"small_model_species_labels: {self.small_species_labels}")
        self.big_species_labels: Dict[int, str] = big_model_species_labels
        self.big_species_name = list(self.big_species_labels.values())
        self.is_big_incv3 = is_big_inception_v3
        self.small_species_preds: Dict = {}

        self.global_species_preds: list = []
        self.class_calls: Dict[Any, Any] = {}
        for i in range(8):
            self.class_calls[i] = 0

        self.small_model_call: Dict[Any, Any] = {}
        for i in range(self.number_of_small_models):
            self.small_model_call[i] = 0
        self.sessions: list[InferenceSession] = [
            ort.InferenceSession(
                model_path, session_options, providers=[self.providers[0]]
            )
            for model_path in self.small_model_path
        ]
        self.big_model_call: int = 0
        self.correct_predictions: int = 0

    def _get_small_model_other_label(self, small_species_lab) -> int:
        """
        Get the label for 'Other' species in the small model.
        """
        species_labels_flip: Dict[str, int] = dict(
            (v, k) for k, v in small_species_lab.items()
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
        self, prediction: int, model_type: ModelType, spececies_model_idx: int = 0
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
            # print(f"Big species label: {big_species_label}")
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
            # print(
            #     f"Small model index: {self.small_species_labels[spececies_model_idx]}"
            # )
            small_species_label = self.small_species_labels[spececies_model_idx].get(
                prediction, None
            )
            # print(f"Small species label: {small_species_label}")
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
        # print(f"Total species in global dataset: {(sampled_species)}")
        remaining_k: int = sample_size - len(sampled_species)
        sampled_species += random.choices(
            population=sampled_species,
            weights=[
                self.global_species_probs[sid]
                for sid in self.global_species_labels.keys()
            ],
            k=remaining_k,
        )
        random.shuffle(sampled_species)

        return [label for label in sampled_species]

    def _infer_one(
        self,
        model_type: ModelType,
        image_path: str,
        small_session: Optional[InferenceSession] = None,
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
            session = self.big_session
            input_size = self.big_input_size
            input_name = self.big_input_name
            if self.is_big_incv3:
                is_incv3 = True
            else:
                is_incv3 = False
        else:
            session: InferenceSession = small_session  # type: ignore
            input_size = self.small_input_size
            input_name = session.get_inputs()[0].name  # type: ignore
            is_incv3 = False

        try:
            img = preprocess_eval_opencv(
                image_path, *input_size, is_inception_v3=is_incv3
            )
            outputs = session.run(None, {input_name: img})  # type: ignore
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None

    def infer_with_routing(
        self, image_path: str, ground_truth: int
    ) -> Optional[Tuple[str, str, int, float]]:
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

        gt_path = f"{image_path.split('/')[-2]}"
        # small: int = 0
        # create start counter:
        start = time.perf_counter()
        for small in range(self.number_of_small_models):
            small_result = self._infer_one(
                ModelType.SMALL_MODEL, image_path, self.sessions[small]
            )

            if small_result is None:
                print(f"Small model returns no result for {image_path}")
                return None
            small_pred, _ = small_result
            # self.class_calls[small_pred] += 1
            if small_pred == self.small_other_labels[small]:
                # print("try-next-small-model")
                if small == self.number_of_small_models - 1:
                    end = time.perf_counter()
                    small_species_label = "Evergestis pallidata"
                    if small_species_label == gt_path:
                        self.correct_predictions += 1

                    return small_species_label, gt_path, small, end - start  # type: ignore
                # next iteration:
                continue
            else:
                end = time.perf_counter()
                # print("Predict complete, next image")
                self.small_model_call[small] += 1
                # get the name of the predicted species:
                small_species_label = self.small_species_labels[small].get(
                    small_pred, None
                )
                # print(f"ground truth path: {gt_path}")
                if small_species_label == gt_path:
                    self.correct_predictions += 1
                # Translate the small model prediction to the global species label
                return small_species_label, gt_path, small, end - start  # type: ignore

        # small += 1
        # print(f"All small models returned 'Other', running big model for {image_path}")
        # big_result = self._infer_one(ModelType.BIG_MODEL, image_path)

        # print(f"ground truth path: {gt_path}"
        # if big_result is None:
        #     print(f"Big model returns no result for {image_path}")
        #     return None
        # end = time.perf_counter()
        # self.big_model_call += 1
        # big_species_label = self.global_species_labels.get(big_result[0], None)
        # if big_species_label == gt_path:
        #     self.correct_predictions += 1
        # return big_species_label, gt_path, self.number_of_small_models, end - start  # type: ignore

    def run(
        self,
        num_runs: int,
        sample_size: int = 1000,
        save_path=None,
        # model_type: str = "both",
    ) -> None:
        other_preds = 0
        y_true: List[int] = []
        y_pred: List[int] = []
        FNs = {}
        for i in range(self.number_of_small_models):
            FNs[i] = 0
        latencies = {}
        for i in range(self.number_of_small_models + 1):
            latencies[i] = []
        time_per_image = []

        for image_path in tqdm(self.small_data_manifests):
            start = time.perf_counter()
            result = self.infer_with_routing(image_path, 0)
            end = time.perf_counter()

            if result is None:
                print(f"Model returns no result for {image_path}")
                continue

            if result is not None:
                pred, gt, small_counter, latency = result
                latencies[small_counter].append(latency)
                time_per_image.append(end - start)
                if small_counter < self.number_of_small_models:
                    if calculate_FN(gt, pred, self.small_species_labels, small_counter):
                        FNs[small_counter] += 1

                gt_int = [
                    key for key, val in self.global_species_labels.items() if val == gt
                ]

                pred_int = [
                    key
                    for key, val in self.global_species_labels.items()
                    if val == pred
                ]

                y_true.append(gt_int[0])
                y_pred.append(pred_int[0])

        total_support_list = get_support_list(
            self.global_total_support_list, self.global_species_names
        )
        for i in range(self.number_of_small_models):
            FNs[i] /= 1000

        print(f"Total 'Other' prediction by small model: {other_preds}")
        # print(f"Total prediction outside the test set: {num_pred_outside_global}")

        print(f"Number calls of each class: {self.class_calls}")
        print(f"Small model calls: {self.small_model_call}")
        print(f"Big model call: {self.big_model_call}/{len(self.small_data_manifests)}")
        print(
            f"Correct predictions: {self.correct_predictions}/{len(self.small_data_manifests)}"
        )
        print(
            f"accuracy: {self.correct_predictions / len(self.small_data_manifests):.4f}"
        )
        print(f"Average time per image: {stats.fmean(time_per_image) * 1000:.2f} ms")
        print(f"Throughput: {1.0 / stats.fmean(time_per_image):.2f} fps")
        # total_support_list.append(num_pred_outside_global)
        for i in range(self.number_of_small_models):
            print(
                f"Average time per image with small model {i}: {stats.fmean(latencies[i]) * 1000:.2f} ms"
            )
        # print(
        #     f"Average time per image with large model: {self.number_of_small_models} {stats.fmean(latencies[self.number_of_small_models]) * 1000:.2f} ms"
        # )
        if save_path:
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"False negative rate of model: {FNs}")

            df = generate_report(
                y_true,
                y_pred,
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

    arg_parser.add_argument(
        "--n_mcunets", type=int, default=1, help="Number of mcunets used"
    )

    arg_parser.add_argument(
        "--num_of_dominant_classes",
        type=int,
        default=13,
        help="Number of dominant classes used",
    )
    # arg_parser.add_argument("--save_path", type=str, help="Evaluation directory")
    args = arg_parser.parse_args()

    threshold = args.threshold
    small_image_data = []
    small_species_labels = []
    # small_model_paths = [
    #     f"models/mcunet-in2_haute_garonne_0.5_{i}_best.onnx" for i in range(0, 15, 7)
    # ]
    # small_image_data, _, _, small_species_labels, _ = manifest_generator_wrapper(
    #     threshold
    # )
    num_of_dominant_classes = args.num_of_dominant_classes
    data_creators = DatasetCreator(number_of_dominant_classes=num_of_dominant_classes)
    n_mcunets = args.n_mcunets
    n = 1 + num_of_dominant_classes * (n_mcunets - 1)
    label_counts = {}
    counter = 0
    sample_count = 0
    dominant_threshold = 0.6
    test_image_path = []
    for i in range(0, n, num_of_dominant_classes):
        small_image_dats, _, _, _, small_species_label = data_creators.create_dataset(i)
        # print(f"small species label: {small_species_label}")

        for d in small_image_dats:
            label = d["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        print(label_counts)

        # save image path in small image dats into a json file
        for image_label_path in small_image_dats:
            if sample_count < 1000:
                test_image_path.append(image_label_path["image"])
                if image_label_path["label"] < num_of_dominant_classes:
                    counter += 1
                sample_count += 1

        small_image_data.append(small_image_dats)
        small_species_labels.append(small_species_label)
    print(f"counter: {counter}")
    print(f"Dataset Ratio: {counter / len(test_image_path)}")
    # print(f"Typ e of global image data: {type(global_image_data)}")
    global_image_data, _, _, global_species_labels, global_species_composition = (
        manifest_generator_wrapper(1.0)
    )
    # dump to json file
    print(f"global species label: {global_species_labels}")
    with open("haute_garonne/dataset_species_labels.json") as full_bird_insect_labels:
        big_species_labels = json.load(full_bird_insect_labels)
        big_species_labels = {int(k): v for k, v in big_species_labels.items()}
    # Create a list of small model paths
    small_model_paths = [
        f"models/mcunet-in2_haute_garonne_{dominant_threshold}_{i}_best.onnx"
        for i in range(0, n, num_of_dominant_classes)
    ]
    big_model_path = "models/convnext_full_insect_best.onnx"
    pipeline = NFullPipelineMonteCarloSimulation(
        small_model_paths,
        big_model_path,
        test_image_path,
        global_image_data,
        global_species_labels,
        global_species_composition,
        small_species_labels,
        big_species_labels,
        is_big_inception_v3=False,
        small_model_input_size=args.small_model_input_size,
        big_model_input_size=args.big_model_input_size,
        # providers=["CUDAExecutionProvider", "CUDAExecutionProvider"],
    )

    pipeline.run(1, 1000, args.save_path)
