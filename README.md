# nuLSQ-new

### How to run
- Modify the experiment parameters in configs.py before training.
- Place the pre-trained model in the model_zoo folder.
- Run by `python main.py`.
- The results will be saved at work_dir.

### To do list
- Need to implement progressive learning support
- Need to implement the logging function to record how the step size change during training.

### Pre-trained Model
- Currently only support PreResNet-18 for imagenet training and PreResNet-20 for CIFAR100 training. (New model can be added later.)
- Link for floating-point pre-trained model. ([model link](https://drive.google.com/drive/folders/1CPhGPSlAbwA1irpjcr2S_VB-vsn5FZtQ?usp=sharing))