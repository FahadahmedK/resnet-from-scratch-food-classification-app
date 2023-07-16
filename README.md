# resnet-from-scratch-for-hotdog-classification-app
Demo application to classify hotdog (or not) using ResNet from scratch. The repo also shows how to leverage MLFlow and Kubeflow for ML experiment tracking and orchestration.



# how to organize your data:
```
from pathlib import Path
import json
with open('../data/meta/train.json', 'r') as file:
    train_meta  = json.load(file)
with open('../data/meta/test.json', 'r') as file:
    test_meta  = json.load(file)
test_data_paths = [Path(value+'.jpg') for key, values in test_meta.items() for value in values]

train_path = Path('../data/images/train')
test_path = Path('../data/images/test')
Path(train_path).mkdir(parents=True, exist_ok=True)
Path(test_path).mkdir(parents=True, exist_ok=True)
for key, values in train_meta.items():
    for value in values:
        (train_path/value).parent.mkdir(parents=True, exist_ok=True)
        (train_path.parent/(value+'.jpg')).rename(train_path/(value+'.jpg'))
for key, values in test_meta.items():
    for value in values:
        (test_path/value).parent.mkdir(parents=True, exist_ok=True)
        (test_path.parent/(value+'.jpg')).rename(test_path/(value+'.jpg'))
```