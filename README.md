# Dataset Library for 3D Machine Learning

This Python dataset library is designed to streamline the end-to-end model training process, enabling efficient loading, visualization, and preparation of 3D data for machine learning applications. It supports advanced techniques, including graph neural networks and voxelized methods, with seamless integration into PyTorch workflows.

## Features

- **Data Storage:** Organized in folders containing `.cgns` files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**
  - **DGL Support:**
    - Surface and volume data in mesh format.
    - 3D visualization of samples and predictions.
    - L2 loss computation and aggregate force evaluation for model training.
  - **Planned PyG Support:** Implementing functionalities similar to DGL.
- **Hugging Face Integration:** Direct dataset loading from [Hugging Face](https://huggingface.co/).
- **Voxelized Flow Field Support:** Facilitates image processing-based ML approaches (Planned).
- **Scalable Data Handling:** Support for larger datasets through the [TUM Library](https://www.ub.tum.de/en/research-data) (planned)
- **Comprehensive Metadata Accessibility:** For advanced model comparison and evaluation (Planned).

## Installation

Currently you need to clone the repository and add it to your Python path. We are working on making it available through `pip`.

## Example Usage

See the `examples` folder for a detailed example of how to use the library.

## Roadmap

- DGL Support
- PyG Support
- Re-meshing with Random Point Sampling
- Voxelized Flow Field Support
- Inference of Surface Quantities from Volumetric Predictions
- Enhanced Metadata Accessibility

## Development

This package uses [uv](https://docs.astral.sh/uv/) for package management. To get started, first install uv. Then run

```bash
uv venv
uv sync
```
to create a virtualenv and install the required dependencies in it. For dgl, run the [install script](examples/meshgraphnet/install.sh).
