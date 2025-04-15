from pipeline.comparing import extracting_feature_maps
from pipeline.utility import mobile_net_v3_large_builder, get_device

device = get_device(use_cpu=True)
model = mobile_net_v3_large_builder(device, path="./models/mobilenet_v3_large_50.pth")  

df_features = extracting_feature_maps(model)
print(df_features)
