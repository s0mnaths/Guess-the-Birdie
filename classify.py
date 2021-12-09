import torch
import torchvision.transforms as tt
from model import WhatBirdie
import pandas as pd

def classifyImages(image, weights_file):
    # Load the model
    
    model = WhatBirdie()
    model.load_state_dict(torch.load('weights91.pth', map_location='cpu'))
    
    # Transform the image to feed into the model (ToTensor, normalize)
    x, y = image.size
    maxsize = max(x, y)

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = tt.Compose([tt.Pad(maxsize, padding_mode='reflect'),
                            tt.CenterCrop(maxsize),
                            tt.Resize(224),
                            tt.ToTensor(),
                            tt.Normalize(*stats,inplace=True)
                            ])

    test_image = torch.unsqueeze(transform(image), 0)


    # Predict
    output = model(test_image)
    _, preds  = torch.max(output, dim=1)   # torch.max -> Pick index with highest probability

    # Retrieve the class label
    return preds[0].item()


# Getting the class label
species_list = pd.read_csv("species.csv")