# Pretrained Models
A repository of pretrained multiples required for reproducing experiments 

## Torch Instructions
  1. Save pretrained models using directly using `torch.save(model, path)`. 
  2. Push `path` to this guthub repository
  3. Load and run models using `torch.utils.model_zoo.load_url(url_for_raw_file)` 

See example below:

```python
import torch
from torchvision.datasets import CIFAR10

# download model
model = torch.utils.model_zoo.load_url(
        'https://github.com/rgklab/pretrained_models/blob/main/detectron/cifar10.pt?raw=true'
      ).cuda()

# download transform
tf = torch.utils.model_zoo.load_url(
      'https://github.com/rgklab/pretrained_models/blob/main/detectron/cifar10_input_transform.pt?raw=true'
    )
data = CIFAR10('/voyager/datasets', transform=tf)

model.eval()
logits = model(data[0][0].unsqueeze(0).cuda())[0]
print(f'Prediction: {data.classes[logits.argmax()]}, Probability: {100*logits.softmax(0).max():.2f}, True: {data.classes[data[0][1]]}')
## Prediction: frog, Probability: 97.58, True: frog
```
