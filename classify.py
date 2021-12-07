import torch
import torchvision.transforms as tt
from model import WhatBirdie

def classifyImages(image, weights_file):
    # Load the model
    
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WhatBirdie()
    model.load_state_dict(torch.load('weights91.pth', map_location='cpu'))

            # model = torch.load("weights91.pth")
    
    # Transform the image to feed into the model (ToTensor, normalize)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = tt.Compose([tt.Resize(256),
                            tt.CenterCrop(224),
                            tt.ToTensor(),
                            tt.Normalize(*stats,inplace=True)
                            ])

    test_image = torch.unsqueeze(transform(image), 0)
            # test_image = test_image.to(device)


    # Predict
    output = model(test_image)
    _, preds  = torch.max(output, dim=1)   # torch.max -> Pick index with highest probability

    # Retrieve the class label
    return preds[0].item()