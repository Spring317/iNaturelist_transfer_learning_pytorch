import json
import pandas as pd
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import math

def load_class_counts(json_path):
    """Load class counts from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['class_counts_sorted']

def get_available_classes(dataset_path):
    """Get list of available classes in the dataset"""
    insecta_path = os.path.join(dataset_path, 'Insecta')
    if not os.path.exists(insecta_path):
        return []
    
    classes = []
    for item in os.listdir(insecta_path):
        class_path = os.path.join(insecta_path, item)
        if os.path.isdir(class_path):
            classes.append(item)
    return classes

def get_class_images(dataset_path, class_name):
    """Get all image files for a specific class"""
    class_path = os.path.join(dataset_path, 'Insecta', class_name)
    if not os.path.exists(class_path):
        return []
    
    images = []
    for file in os.listdir(class_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            images.append(os.path.join(class_path, file))
    return images

def calculate_samples_per_class(class_counts, total_samples, class_indices):
    """Calculate number of samples per class respecting the ratio"""
    # Get subset of classes
    subset_classes = [class_counts[i] for i in class_indices]
    
    # Calculate total count for this subset
    total_count = sum(cls['sample_count'] for cls in subset_classes)
    
    # Calculate samples per class maintaining ratio
    samples_per_class = {}
    allocated_samples = 0
    
    for i, cls in enumerate(subset_classes):
        if i == len(subset_classes) - 1:  # Last class gets remaining samples
            samples = total_samples - allocated_samples
        else:
            ratio = cls['sample_count'] / total_count
            samples = max(1, round(ratio * total_samples))  # At least 1 sample per class
        
        samples_per_class[cls['class_name']] = samples
        allocated_samples += samples
    
    return samples_per_class

def create_test_set(dataset_path, class_counts, distribution_config, output_path, test_name):
    """Create a test set based on distribution configuration"""
    print(f"\nCreating {test_name}...")
    
    # Get available classes
    available_classes = get_available_classes(dataset_path)
    print(f"Available classes in dataset: {len(available_classes)}")
    
    # Check for specific class
    if "Evergestis pallidata" in available_classes:
        print("✓ Evergestis pallidata found in dataset")
    else:
        print("✗ Evergestis pallidata NOT found in dataset")
    
    # Filter class_counts to only include available classes
    filtered_class_counts = []
    missing_classes = []
    for cls in class_counts:
        if cls['class_name'] in available_classes:
            filtered_class_counts.append(cls)
        else:
            missing_classes.append(cls['class_name'])
    
    print(f"Total available classes: {len(filtered_class_counts)}")
    print(f"Missing classes from dataset: {len(missing_classes)}")
    
    # Check if Evergestis pallidata is in class_counts but missing from dataset
    evergestis_in_counts = any(cls['class_name'] == "Evergestis pallidata" for cls in class_counts)
    if evergestis_in_counts and "Evergestis pallidata" in missing_classes:
        print("! Evergestis pallidata is in class_counts but missing from dataset directory")
    elif not evergestis_in_counts:
        print("! Evergestis pallidata is not in class_counts_sorted.json")
    
    # Create output directory
    test_output_path = os.path.join(output_path, test_name)
    insecta_output_path = os.path.join(test_output_path, 'Insecta')
    os.makedirs(insecta_output_path, exist_ok=True)
    
    total_collected = 0
    class_distribution = {}
    
    # Process each tier in the distribution
    current_class_idx = 0
    
    for tier_name, (num_classes, tier_samples) in distribution_config.items():
        print(f"\nProcessing {tier_name}: {num_classes} classes, {tier_samples} samples")
        
        # Determine class indices for this tier
        end_idx = min(current_class_idx + num_classes, len(filtered_class_counts))
        class_indices = list(range(current_class_idx, end_idx))
        
        if not class_indices:
            print(f"No more classes available for {tier_name}")
            break
        
        # Calculate samples per class for this tier
        samples_per_class = calculate_samples_per_class(filtered_class_counts, tier_samples, class_indices)
        
        # Collect samples for each class in this tier
        tier_collected = 0
        for class_idx in class_indices:
            class_info = filtered_class_counts[class_idx]
            class_name = class_info['class_name']
            target_samples = samples_per_class[class_name]
            
            # Get available images for this class
            available_images = get_class_images(dataset_path, class_name)
            
            if not available_images:
                print(f"Warning: No images found for class {class_name}")
                continue
            
            # Randomly sample images (without replacement)
            actual_samples = min(target_samples, len(available_images))
            selected_images = random.sample(available_images, actual_samples)
            
            # Create class directory in output
            class_output_dir = os.path.join(insecta_output_path, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Copy selected images
            for i, img_path in enumerate(selected_images):
                filename = f"{class_name}_{i+1:04d}.jpg"
                output_img_path = os.path.join(class_output_dir, filename)
                shutil.copy2(img_path, output_img_path)
            
            class_distribution[class_name] = {
                'target_samples': target_samples,
                'actual_samples': actual_samples,
                'available_samples': len(available_images),
                'tier': tier_name,
                'rank': class_info['rank']
            }
            
            tier_collected += actual_samples
            total_collected += actual_samples
            
            print(f"  {class_name}: {actual_samples}/{target_samples} samples (rank {class_info['rank']})")
        
        print(f"{tier_name} total: {tier_collected} samples")
        current_class_idx = end_idx
    
    # Save distribution info with debugging information
    distribution_info = {
        'test_name': test_name,
        'total_samples': total_collected,
        'total_classes': len(class_distribution),
        'distribution_config': distribution_config,
        'class_distribution': class_distribution,
        'debug_info': {
            'total_classes_in_counts': len(class_counts),
            'available_classes_in_dataset': len(available_classes),
            'filtered_classes_used': len(filtered_class_counts),
            'missing_classes_count': len(missing_classes),
            'missing_classes': missing_classes[:20]  # Save first 20 missing classes
        }
    }
    
    with open(os.path.join(test_output_path, 'distribution_info.json'), 'w') as f:
        json.dump(distribution_info, f, indent=2)
    
    print(f"\n{test_name} completed: {total_collected} samples, {len(class_distribution)} classes")
    return distribution_info

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    dataset_path = '/home/quydx/iNaturelist_transfer_learning_pytorch/haute_garonne'
    class_counts_path = '/home/quydx/iNaturelist_transfer_learning_pytorch/data_prep/class_counts_sorted.json'
    output_path = '/home/quydx/iNaturelist_transfer_learning_pytorch/test_sets'
    
    # Load class counts
    class_counts = load_class_counts(class_counts_path)
    print(f"Loaded {len(class_counts)} classes from class_counts_sorted.json")
    
    # Check if Evergestis pallidata is in class_counts
    evergestis_in_counts = any(cls['class_name'] == "Evergestis pallidata" for cls in class_counts)
    if evergestis_in_counts:
        evergestis_info = next(cls for cls in class_counts if cls['class_name'] == "Evergestis pallidata")
        print(f"✓ Evergestis pallidata found in class_counts at rank {evergestis_info['rank']} with {evergestis_info['sample_count']} samples")
    else:
        print("✗ Evergestis pallidata NOT found in class_counts_sorted.json")
    
    # Check if class exists in dataset
    dataset_insecta_path = os.path.join(dataset_path, 'Insecta')
    evergestis_path = os.path.join(dataset_insecta_path, 'Evergestis pallidata')
    if os.path.exists(evergestis_path):
        images = get_class_images(dataset_path, 'Evergestis pallidata')
        print(f"✓ Evergestis pallidata directory found with {len(images)} images")
    else:
        print("✗ Evergestis pallidata directory NOT found in dataset")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Define distribution configurations
    distributions = {
        'DI1': {
            'tier1_top13': (13, 800),
            'tier2_remaining': (1000, 200)  # Use large number to include all remaining
        },
        'DI2': {
            'tier1_top13': (13, 400),
            'tier2_next13': (13, 400),
            'tier3_remaining': (1000, 200)
        },
        'DI3': {
            'tier1_top13': (13, 300),
            'tier2_next13': (13, 300),
            'tier3_next13': (13, 200),
            'tier4_remaining': (1000, 200)
        }
    }
    
    # Create test sets
    results = {}
    for test_name, config in distributions.items():
        try:
            result = create_test_set(dataset_path, class_counts, config, output_path, test_name)
            results[test_name] = result
        except Exception as e:
            print(f"Error creating {test_name}: {str(e)}")
    
    # Save overall summary
    summary = {
        'creation_date': pd.Timestamp.now().isoformat(),
        'source_dataset': dataset_path,
        'class_counts_source': class_counts_path,
        'random_seed': 42,
        'results': results
    }
    
    with open(os.path.join(output_path, 'test_sets_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("TEST SETS CREATION SUMMARY")
    print("="*50)
    for test_name, result in results.items():
        print(f"{test_name}: {result['total_samples']} samples, {result['total_classes']} classes")
    
    print(f"\nAll test sets saved to: {output_path}")

if __name__ == "__main__":
    main()