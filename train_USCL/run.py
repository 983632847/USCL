from simclr import SimCLR
import yaml
import random
import numpy as np
import math

from data_aug.dataset_wrapper_Ultrasound_Video_Mixup import DataSetWrapper     # Video_Mixup


def main():
    # Totalcases = 1051    # US-4
    # Totalcases = 63      # CLUST
    # Totalcases = 296     # Liver
    Totalcases = 22      # Butterfly
    # Totalcases = 670     # COVID19


    # LabelRate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    LabelRate = [1]

    # DataRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    DataRate = [1]

    Checkpoint_Num = 1
    for LL in range(len(LabelRate)):
        for DD in range(len(DataRate)):
            LabelList = random.sample(range(0, Totalcases), math.ceil(Totalcases * LabelRate[LL]))
            DataList = random.sample(range(0, Totalcases), math.ceil(Totalcases * DataRate[DD]))
            config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
            dataset = DataSetWrapper(config['batch_size'], LabelList, DataList, Checkpoint_Num, **config['dataset'])
            lumbda = [0.1, 0.2, 0.3, 0.4, 0.5]  # Semi-supervised CL, default=0.2
            for i in range(len(lumbda)):
                simclr = SimCLR(dataset, config, lumbda[i], Checkpoint_Num)
                simclr.train()
                Checkpoint_Num += 1

if __name__ == "__main__":
    main()
