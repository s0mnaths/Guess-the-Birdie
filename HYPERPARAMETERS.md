| Accuracy | Time | Transforms | Batch Size | Batch Norm layer | Epochs | lr_Scheduling(Max lr) | Weight decay | Gradient clipping | Optimizer |
|----------|------|------------|------------|------------------|--------|-----------------------|--------------|-------------------|-----------|
|  59 %  | -  | - |  128 |  - |  10 |  - | -  | -  | Adam  |
|  61 % | -  | - |  256 |  - |  10 |  - | -  | -  | Adam  |  
|  64 % | 21m  | Norm |  256 |  - |  10 |  - | -  | -  | Adam  |  
|  58 % | 13m  | Norm, scale(56) |  256 |  - |  10 |  - | -  | -  | Adam  | 
|  72 % | 13m  | Norm, randomcrop, randomhorizontalflip |  256 |  - |  10 |  - | -  | -  | Adam  | 