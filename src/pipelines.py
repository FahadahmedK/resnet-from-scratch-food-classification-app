import shutil
import kfp


# For creating the pipeline
from kfp import dsl

# For building components
from kfp.dsl import component
from kubernetes.client import V1VolumeMount, V1Volume

# Type annotations for the component artifacts
from kfp.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
    InputPath,
    OutputPath,
)
#from preprocess import preprocess_images


#@component(
#    packages_to_install=["torchvision", "Pillow", "click", "torch", "numpy"],
#    output_component_file="preprocessing_component.yaml",
#)
#def preprocessing_op(
#    data_dir: InputPath(str),
#    output_dir: InputPath(str),
#    processed_data_artifact: Output[Artifact]
#):
 #   import os
 #   import numpy as np
 #   from torchvision import transforms
 #   from torch.utils.data import DataLoader, Dataset
 #   from torchvision.utils import save_image
 #   preprocess_images(data_dir, output_dir, batch_size=32)
  #  return output_dir

#@dsl.pipeline(
#    name="ML Pipeline",
#    description="Pipeline for image classification"
#)
#def ml_pipeline(
#    data_dir: str,
#    output_dir: str,
#):
#    preprocessing_task = preprocessing_op(data_dir=data_dir, output_dir=output_dir)


@component(
    packages_to_install=["pandas", "openpyxl"],
)

def download_op(url:str, local_path: str, output_csv:Output[Dataset]):
    
    import pandas as pd

    # Use pandas excel reader
    df = pd.read_excel(url)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_csv.path, index=False)
    
    
    df.to_csv(local_path, index=False)


@dsl.pipeline(
    name="ML Pipeline",
    description="Pipeline for image classification"
)
def ml_pipeline(url: str, local_path: str):
    
    #local_volume = dsl.VolumeOp(
    #    name="local-storage",
    #    resource_name="local-storage",
    #    size="1Gi",
    #    storage_class="local-storage",
    #   volume='hostPath',
    #    path=local_path
    #)


    download_task = download_op(url=url, local_path=local_path).add_pvolumes({local_path: dsl.PipelineVolume(pvc="local-storage")})


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="configs/pipeline.yaml")
