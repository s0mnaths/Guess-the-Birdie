from .model import frcnn
import torch
import torchvision.transforms as tt
import numpy as np
from PIL import Image

def detectBird(image):
    
    transform = tt.ToTensor()
    
    tensorImg = torch.unsqueeze(transform(image), 0)
    frcnn.load_state_dict(torch.load('checkpoints/frcnn50_checkpoint.pth', map_location='cpu'))
    frcnn.eval()
    output = frcnn(tensorImg)

    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    scores = output[0]['scores'].detach().numpy()

    max_prob = max(scores)

    for i in range(len(labels)):
        if (labels[i] == 16) & (scores[i] == max_prob):
            bbox = boxes[i]
            break

    xmin, ymin, xmax, ymax = bbox
    numpyImg = np.asarray(image)
    cropImg = numpyImg[int(ymin)-30:int(ymax)+30, int(xmin)-30:int(xmax)+30]
    finalImg = Image.fromarray(cropImg)

    return finalImg