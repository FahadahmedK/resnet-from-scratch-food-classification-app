from kfp import dsl
from kfp.dsl import component, pipeline
from kfp.components import InputPath, OutputPath
from preprocess import preprocess

@dsl.component(
    packages_to_install=["torchvision", "Pillow", "click", "torch", "numpy"],
    output_component_file="preprocessing_component.yaml",
)
def preprocessing_op(
    data_dir: InputPath("Raw Images Directory"),
    output_dir: OutputPath("Processed Images Directory"),
    batch_size: int = 32
) -> OutputPath("Processed Images Directory"):
    import os
    import numpy as np
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from torchvision.utils import save_image
    preprocess(data_dir, output_dir)


@pipeline(
    name="ML Pipeline",
    description="Pipeline for image classification"
)
def ml_pipeline(
    data_dir: str,
    output_dir: str,
):
    preprocessing_task = preprocessing_op(data_dir, output_dir)


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="pipeline.yaml")