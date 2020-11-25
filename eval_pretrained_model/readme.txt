PyTorch implementation of "Effective Sample Pair Generation for Ultrasound Video Contrastive Representation Learning"

This is a demo to run the pretrained model on POCUS classification dataset. 

1. Pick a model and its config file, for example, config.yaml

2. Download the model best_model.pth

3. Download the 5 fold cross validation POCUS dataset

4. Run the demo with:
    python eval_pretrained_model.py

You will get a total accuracy around 95.0%. The result will fluctuate slightly on different environments.

Our environment is:
Python 3.6.9, Pytorch 1.6.0, CUDA 10.2, Intel Xeon Silver 4210R CPU@2.4GHz, Nvidia Tesla V100 GPU.






