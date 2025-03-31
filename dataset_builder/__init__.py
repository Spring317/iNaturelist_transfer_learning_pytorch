"""
dataset_builder - A package for species dataset preparation and visualization.

This package provides tools to automate the preparation of species datasets by 
extracting, cleaning, and visualizing data from various sources. It is designed 
specifically to work with the iNaturalist dataset but can be easily modified 
to work with other datasets. The package includes functionality for cross-referencing 
species data between multiple datasets, generating visualizations like bar charts 
and cumulative distribution functions (CDF/PPF), and saving datasets in formats 
suitable for machine learning tasks.

Modules:
- analysis: Functions for dataset analysis and cross-referencing between species datasets.
    - run_analyze_dataset: Analyzes species data, extracting species lists and image counts.
    - run_cross_reference: Cross-references species between two datasets, identifying matched and unmatched species.
- builder: Functions for managing dataset files and copying species data between source and destination directories.
    - run_copy_matched_species: Copies species data between datasets based on a matching criteria.
    - run_web_crawl: Scrapes species data from web pages and saves it as JSON.
- manifest: Functions to create dataset manifests, including dominant species identification and saving data to files.
    - run_manifest_generator: Generates and saves dataset manifests, splitting data into training and validation sets.
- visualization: Functions for generating visual representations of species data.
    - run_visualization: Generates visualizations such as species distribution bar charts and PPF plots.
    - venn_diagram: Creates a Venn diagram showing the overlap between two datasets.

Exceptions:
- PipelineError: Base class for all pipeline-related exceptions.
- FailedOperation: Raised when a specific operation in the pipeline fails.
- ConfigError: Raised when configuration is invalid or missing.
- AnalysisError: Raised when an analysis operation fails.
"""

from .analysis import run_analyze_dataset, run_cross_reference
from .builder import run_copy_matched_species, run_web_crawl
from .core import load_config, validate_config
from .manifest import run_manifest_generator
from .visualization import run_visualization, venn_diagram