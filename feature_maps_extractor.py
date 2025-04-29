from pipeline.comparing import extracting_feature_maps
from pipeline.utility import mobile_net_v3_large_builder, get_device
import os

device = get_device(use_cpu=True)
for model_file in os.listdir("/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_global_isomorphic/"):
    if not model_file.endswith("pth"):
        continue
    print(model_file)
    model_path = f"/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_global_isomorphic/{model_file}"
    model_name = model_file.replace(".pth", ".csv")
    model = mobile_net_v3_large_builder(device, path=model_path)  

    df_features = extracting_feature_maps(model)

    df_features.to_csv(f"feature_maps/{model_name}")
# model_path = "./model/new_model/mobilenet_v3_large_90_prune_30.pth"
# model = mobile_net_v3_large_builder(device, path=model_path)  

# df_features = extracting_feature_maps(model)

# df_features.to_csv(os.path.basename(model_path).replace(".pth", ".csv"))