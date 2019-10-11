# retina-fundus-segmentation

Retina fundus segmentation using U-net on the [DRIVE dataset](https://www.isi.uu.nl/Research/Databases/DRIVE/).

A pretrained model can be found [here](https://unioxfordnexus-my.sharepoint.com/:f:/g/personal/engs1889_ox_ac_uk/El-RgyHPTCtIrkB3UiPIggMBRsWooOdb7owKhKsEIVnijQ?e=Uljsiz).

To train a model first run `create_dataset.py`. You need to change the path to the raw DRIVE dataset, then run `train.py`. You can adjust any parameters in the function `main` since the data is linked using [fire](https://github.com/google/python-fire).

The results are shown in the notebook `Results.ipynb`.
