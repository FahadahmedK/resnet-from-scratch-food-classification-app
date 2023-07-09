from kfp import dsl
from kfp.dsl import component, pipeline, Artifact, Output
from preprocess import preprocess_images

@component(
    packages_to_install=["torchvision", "Pillow", "click", "torch", "numpy"],
    output_component_file="preprocessing_component.yaml",
)
def preprocessing_op(
    data_dir: str,
    output_dir: str,
    processed_data_artifact: Output[Artifact]
):
    import os
    import numpy as np
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from torchvision.utils import save_image
    preprocess_images(data_dir, output_dir, batch_size=32)
    return output_dir

@pipeline(
    name="ML Pipeline",
    description="Pipeline for image classification"
)
def ml_pipeline(
    data_dir: str,
    output_dir: str,
):
    preprocessing_task = preprocessing_op(data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="configs/pipeline.yaml")
