import torch
import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import manifest_generator_wrapper, get_support_list, generate_report
from typing import Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    checked_labels = False
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # tensor guard
        if not checked_labels:
            num_classes = model(images).shape[1]
            label_min = labels.min().item()
            label_max = labels.max().item()

            if labels.min() < 0 or labels.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected!\n"
                    f"Labels: {labels}\n"
                    f"Min: {label_min}, Max: {label_max}\n"
                    f"Model output classes: {num_classes}"
                )
            checked_labels = True

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.detach().item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def train_validate(
    model: torch.nn.Module,
    dataloader: DataLoader[CustomDataset],
    criterion,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.detach().item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    return avg_loss, accuracy, float(macro_f1)


def save_model(
    model: torch.nn.Module, 
    name: str, 
    save_path: str, 
    device: torch.device, 
    img_size: Tuple[int ,int]
):
    os.makedirs(save_path, exist_ok=True)
    pytorch_path = os.path.join(save_path, f"{name}.pth")
    torch.save(model, pytorch_path)
    print(f"Saved Pytorch model to {pytorch_path}")

    dummy_input = torch.randn(1, 3, *img_size, device=device)
    onnx_path = os.path.join(save_path, f"{name}.onnx")
    torch.onnx.export(
        model,
        (dummy_input, ),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {onnx_path}")


def validation_onnx(onnx_path, val_loader, species_names, device="cpu"):
    model_name = os.path.basename(onnx_path)
    dom = model_name.split("_")[1]
    manifest_generator_wrapper(dom/100)
    total_support_list = get_support_list("./data/haute_garonne/species_composition.json", species_names)

    ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    all_preds, all_labels = [], []
    

    for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
        # Convert tensor to numpy + permute to NHWC
        images_np = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

        outputs = ort_session.run([output_name], {input_name: images_np})[0]
        preds = np.argmax(outputs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"ONNX Validation Accuracy: {accuracy:.4f}")
    print(f"ONNX Weighted Recall: {recall:.4f}")
    print(f"ONNX Weighted F1-Score: {f1:.4f}")

    report_df = generate_report(all_labels, all_preds, species_names, total_support_list, float(accuracy))
    return report_df