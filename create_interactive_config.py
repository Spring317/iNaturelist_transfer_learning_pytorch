from dataset_builder.core import build_interactive_config, save_config
config = build_interactive_config()
save_config(config, "config.yaml")