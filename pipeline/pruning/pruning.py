import torch
import torch_pruning as tp


def one_shot_prune(
    model: torch.nn.Module, 
    importance,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    ignored_layers,
) -> torch.nn.Module:
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,  
        iterative_steps=1,  
        ignored_layers=ignored_layers,
        global_pruning=False,
    )
    pruner.step()
    return model