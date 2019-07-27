## Implementation of ShuflfleNetV2 + SSD

### results

- trainset

  VOC 2007 + VOC2012

- test

  VOC2007

- result

  mAp: 0.65

  

### command

- train

```
python train_pelee.py --dataset VOC --config ./configs/Pelee_VOC.py
```

- test on voc

  ```
  python test.py --dataset VOC\COCO  --config ./configs/Pelee_VOC.py --trained_model ./weights/Pelee_VOC.pth 
  ```

- test your own pic

  ```
  python demo.py --dataset VOC\COCO  --config ./configs/Pelee_VOC.py --trained_model ./weights/Pelee_VOC.pth --show
  ```

### to do

get dataset from [pytorch.ssd](<https://github.com/amdegroot/ssd.pytorch> )

