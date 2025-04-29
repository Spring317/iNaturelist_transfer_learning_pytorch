import random
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import scipy.special
from tqdm import tqdm
from typing import Dict, List, Union, Tuple
from utility import preprocess_eval_opencv
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from dataset_builder.core.utility import load_manifest_parquet
from pipeline.utility import generate_report, get_support_list

class MonteCarloSimulation:
    def __init__(
        self, 
        model_path, 
        data_manifest: Union[str, List[Tuple[str, int]]], 
        dataset_species_labels, 
        input_size=(224, 224),
        providers: List[str]=["CPUExecutionProvider"]
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.species_labels: Dict[int, str] = dataset_species_labels
        
        self.other_class_id = int(self._get_other_id())
        if isinstance(data_manifest, str):
            self.data_manifest = load_manifest_parquet(data_manifest)
        else:
            self.data_manifest = data_manifest
        self.species_to_images = defaultdict(list)
        self.species_probs = {}
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        for image_path, species_id in self.data_manifest:
            self.species_to_images[species_id].append(image_path)

        total_images = sum(len(imgs) for imgs in self.species_to_images.values())
        self.species_probs = {
            int(species_id): len(images) / total_images
            for species_id, images in self.species_to_images.items()
        }


    def _get_other_id(self):
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.species_labels.items())
        return species_labels_flip.get("Other", -1)


    def _infer_one(self, image_path: str):
        try:
            img = preprocess_eval_opencv(image_path, *self.input_size)
            outputs = self.session.run(None, {self.input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None


    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(map(int, self.species_labels.keys())))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.species_labels.values()))
        fig, ax = plt.subplots(figsize=(40, 40))
        disp.plot(ax=ax, xticks_rotation=50, cmap="Blues", colorbar=True)
        plt.title("Confusion Matrix (Monte Carlo Simulation)")
        plt.tight_layout()
        plt.savefig(f"MonteCarloConfusionMatrix_{os.path.basename(self.model_path).replace(".onnx", "")}.png")


    def _false_positive_rate(self, y_true, y_pred):
        # true: local / false: communication
        # fp: it should be communicate but model predict as local
        # tn: it should be communicate, model predict as communicate
        fp_count = 0
        tn_count = 0
        for label, predict in zip(y_true, y_pred):
            if label == self.other_class_id:
                if predict != label:
                    fp_count += 1
                else:
                    tn_count += 1
        if (fp_count + tn_count) == 0:
            return 0.0
        return fp_count / (fp_count + tn_count) 

    def run_simulation(
        self,
        species_labels: Dict[int, str],
        species_composition: Dict[str, int],
        num_runs: int=30,
        sample_size: int=1000,
        plot_confusion_matrix: bool=False,
        save_path=None,
    ):
        species_names = list(species_labels.values())
        total_support_list = get_support_list(species_composition, species_names)
        comm_rates: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        for run in range(num_runs):
            y_true: List[int] = []
            y_pred: List[int] = []
            sampled_species = []
            for species in self.species_labels.keys():
                sampled_species.append(species)

            remaining_k = sample_size - len(sampled_species)
            sampled_species += random.choices(
                population=list(self.species_labels.keys()),
                weights=[self.species_probs[int(sid)] for sid in self.species_labels.keys()],
                k=remaining_k
            )
            random.shuffle(sampled_species)

            num_comm = 0
            num_local = 0
            correct = 0

            for species_id in tqdm(sampled_species, desc=f"Run {run + 1}/{num_runs}", leave=False):
                image_list = self.species_to_images[int(species_id)]
                if not image_list:
                    print("No image found")
                    continue
                image_path = random.choice(image_list)
                result = self._infer_one(image_path)
                if result is None:
                    continue
                y_true.append(int(species_id))
                y_pred.append(int(result[0]))
                if result[0] == int(species_id):
                    correct += 1
                top1_idx, top1_prop = result
                if top1_idx == self.other_class_id:
                    num_comm += 1
                else:
                    num_local += 1
            
            total_pred = num_comm + num_local
            comm_rate = num_comm / total_pred if total_pred else 0
            comm_rates.append(comm_rate)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        model_name = os.path.basename(self.model_path).replace(".onnx", "")

        if save_path:
            accuracy = accuracy_score(all_true, all_pred)
            df = generate_report(all_true, all_pred, species_names, total_support_list, float(accuracy)) 

            os.makedirs(save_path, exist_ok=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                df.to_csv(os.path.join(save_path, f"{model_name}.csv"))

        print(f"Average communication for {model_name}: {sum(comm_rates)/len(comm_rates)}", end=" ")
        print(f"FPR: {self._false_positive_rate(all_true, all_pred)}")
        if plot_confusion_matrix:
            self._plot_confusion_matrix(all_true, all_pred)
