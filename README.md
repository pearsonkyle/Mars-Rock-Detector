# Automatic Rock Detection and Mapping from HiRISE Imagery

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Chat_Assistant-yellow)](https://hf.co/chat/assistant/67060a7bc5e1bc78b6870519)

 - Development of an automated algorithm for boulder size measurement on Mars using HiRISE/MRO data
- Creation of interactive application to generate training data efficiently
- Optimization of classification algorithm with the generated data
- Combining digital elevation models with results for correlations between rock size and elevation
- Understanding geological history through analysis of Martian surface features, such as periglacial terrains
- Inferring landscape shaping mechanisms (e.g., freeze-thaw cycles, ice sublimation, and ice flow) using spatial distribution of rocks

![](output/Mars10_Poster.png)
![](output/gradient_distribution_comparison.png)

## Installation

1. `conda create -n rock python=3.9`
2. `conda activate rock`
3. `conda install -c conda-forge netCDF4`
4. `conda install numpy scipy matplotlib pandas`
5. `conda install scikit-learn scikit-image opencv`
6. `pip install glymur tqdm rasterio`
7. `unzip training.zip`

## Evaluation

To find/segment rocks in your image use the script below

`python eval_rock.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2`

![](figures/RandomForest_100_91_whiten_prediction.png)

Use the `--plot` flag on the `eval_rock.py` script to plot results while evaluating. The script can take up to 24 hours on some images and will save a mask of the predicted rock shadows that are used for the analysis below. No fine-tune or parameter tuning is necessary, the algorithm is trained on a variety of images and is robust to different lighting conditions and surface types. The algorithm is also parallelized and can take advantage of multiple cores on your machine.

```
usage: eval_rock.py [-h] [-i IMAGE] [-r RES] [-th THREADS] [-tr TRAIN] [-te TEST] [-o OUTDIR] [-p]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Choose a JPEG2000 image to decode
  -r RES, --res RES     Resolution level to decode (0 is highest resolution)
  -th THREADS, --threads THREADS
                        number of threads for reading in JP2000 image
  -tr TRAIN, --train TRAIN
                        training data for rocks
  -te TEST, --test TEST
                        testing data for rocks
  -o OUTDIR, --outdir OUTDIR
                        Directory to save output images
  -p, --plot            plot results
```

## Density estimate

After evaluating an image the output mask can be used to estimate a rock size distribution. The `mask2density.py` script takes a mask of the rock shadows and calculates the rock size distribution. If you provide the script with a mask it can separate the rock size distribution into different regions of the image.

![](output/PSP_001410_2210_RED_A_01_ORTHO/cdf_rock_density.png)

The data shown above is the same however the bottom plotis in a log scale and shows how the two regions converge at the largest rock sizes. 

```
usage: mask2density.py [-h] [-i IMAGE] [-o OUTDIR] [-th THREADS]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Choose a JPEG2000 image to decode
  -o OUTDIR, --outdir OUTDIR
                        Directory to save outputs
  -th THREADS, --threads THREADS
                        number of threads for reading in JP2000 image
```

## DEM Analysis

The rock shadow segments are used to calculate the rock size distribution using a formalism similar to https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2007JE003065 and https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021EA001959 . The `mask2dem.py` script takes a mask of the rock shadows and calculates the rock size distribution. If the corresponding DEM is provided, the script will also correlate the positions to elevation. 

```
python mask2dem.py 
--mask images/PSP_001410_2210_RED_A_01_ORTHO/mask.png 
--dem images/DTEEC_002175_2210_001410_2210_A01.IMG
```

![](output/PSP_001410_2210_RED_A_01_ORTHO/rock_7735_8325.png)

Brain coral is a terrain on Mars that has a similar appearance to a sorted stone circle on Earth. The rock size distribution can be used to infer the geological history of the terrain. For example, the rock size distribution can be used to infer the history of ice and mechanisms that have shaped the landscape e.g. freeze-thaw cycles, ice sublimation, and ice flow. Brain coral masks can be integrated with our analysis and are generated from https://github.com/pearsonkyle/Mars-Brain-Coral-Network

![](images/PSP_001410_2210_RED_A_01_ORTHO_classifier_mask.png)

That mask is low resolution which is from the classifier algorithm in that repo however full-resolution masks are also available. 

![](output/PSP_001410_2210_RED_A_01_ORTHO/elevation_distribution.png)

~1.5 million rocks go into estimating those distributions. They're computed from many tiles that look like such:

![](output/PSP_001410_2210_RED_A_01_ORTHO/cdf_gradient_distribution.png)

## Make new training data

The `label_maker.py` script is used to create/add training data. It takes a JPEG2000 image as input and then you can use the mouse to label regions of the image as either rocks or not rocks. The script will then save the image patches in a directory called `patches` in the current directory. The patches are saved as 11x11 images. The script will organize them based on the source images and class label. The class directory will contain the training samples along with pickle files storing the positions and image data, separately. This is useful for later when you want to train different models using the curated samples.

![](figures/overview.png)

`python label_maker.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2`

### Keyboard Commands:
  - `n`: next tile in big image
  - `d` : delete last sample
  - `c`: change class
  - `t`: train model
  - `p`: predict rocks in image
  - `i`: clear patches
  - `left click`: create training sample


The label making script is set up to distinguish things that are and are not rocks. You can switch between the two classes by pressing the “c” button. Then a label will appear in the bottom right plot based on which class you are currently labelling. Rocks are green and the background is orange. After making a few hundred labels in each class, you can train a classifier by pressing the “t” button and then look at the terminal to see its accuracy. Afterwards, if you want to visualize some predictions, press the “p” button. A bunch of red squares should pop up. In some cases rocks are only visible in the bottom right plot when you mouse over the region but not the main image due to how it is scaled. The rock algorithm sees something similar to the bottom right plot. The classifier is a random forest algorithm from scikit learn.

```
usage: label_maker.py [-h] [-ws WINDOW] [-os OUTSIZE] [-o OUTDIR] [-n NAME] [-i IMAGE] [-r RES] [-t THREADS] [-re]

optional arguments:
  -h, --help            show this help message and exit
  -ws WINDOW, --window WINDOW
                        size of window to search
  -os OUTSIZE, --outsize OUTSIZE
                        size of output window
  -o OUTDIR, --outdir OUTDIR
                        Directory to save output images
  -n NAME, --name NAME  Name of output images
  -i IMAGE, --image IMAGE
                        Choose a JPEG2000 image to decode
  -r RES, --res RES     Resolution level to decode (0 is highest resolution)
  -t THREADS, --threads THREADS
                        number of threads for background class
  -re, --reset          delete existing labels4
```

After some images are labelled we can export the samples to a CSV file and later compile it into a single file which we can easily share and use for training. To create a CSV file run the `make_csv.py` script. The script will create a CSV file in the current directory with the name `samples.csv`. The CSV file will contain the image path, class label, and position of the sample. The script will also create a directory called `patches` which will contain the training samples. The patches are saved as 11x11 images. The script will organize them based on the source images and class label. The class directory will contain the training samples along with pickle files storing the positions and image data, separately. This is useful for later when you want to train different models using the curated samples.

`python make_csv.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2 -ws 11`

Make sure to run this script on each image you labelled. Afterwards, combine all of the available CSV files into a single one for training.

`python combine_csv.py`


## Image labels

To download the data used in our study, navigate to the `images/` folder and run the download script or download the data manually from the HiRISE website:

https://www.uahirise.org/PDS/DTM/ESP/ORB_037200_037299/ESP_037262_1845_ESP_036761_1845/ESP_037262_1845_RED_A_01_ORTHO.JP2
https://www.uahirise.org/PDS/DTM/ESP/ORB_037200_037299/ESP_037262_1845_ESP_036761_1845/DTEEC_037262_1845_036761_1845_U01.IMG
https://www.uahirise.org/PDS/DTM/PSP/ORB_002100_002199/PSP_002175_2210_PSP_001410_2210/PSP_001410_2210_RED_A_01_ORTHO.JP2
https://www.uahirise.org/PDS/DTM/PSP/ORB_002100_002199/PSP_002175_2210_PSP_001410_2210/DTEEC_002175_2210_001410_2210_A01.IMG

https://hirise.lpl.arizona.edu/dtm/ESP_037262_1845
https://hirise.lpl.arizona.edu/dtm/PSP_001410_2210


## Training

![](figures/rock_roc.png)

After collecting a bunch of training samples we train an ensemble of 10 random forests using 4598 images (11x11) and evaluated on 302 images (102 rock, 200 background). The ROC plot can be reproduced running `trainer.py`.


## Reproducing Results
1. `conda create -n rock python=3.9`
2. `conda activate rock`
3. `conda install numpy scipy matplotlib pandas`
4. `conda install scikit-learn scikit-image opencv`
5. `pip install glymur tqdm`
6. `cd images`
7. `chmod +x batch_download.sh`
8. `./batch_download.sh`
9. `cd ..`
10. `python label_maker.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2`
11. `python make_csv.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2 -ws 11`
12. `python label_maker.py -i images/ESP_037262_1845_RED_A_01_ORTHO.JP2`
13. `python make_csv.py -i images/ESP_037262_1845_RED_A_01_ORTHO.JP2 -ws 11`
14. `python combine_csv.py -ws 11`
15. `python trainer.py -ws 11`
16. `python eval.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2`
17. `python eval.py -i images/ESP_037262_1845_RED_A_01_ORTHO.JP2`
18. `python mask2density.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2`
19. `python mask2density.py -i images/ESP_037262_1845_RED_A_01_ORTHO.JP2`


## re-training on different window sizes
1. `python make_csv.py -i images/PSP_001410_2210_RED_A_01_ORTHO.JP2 -ws 19`
2. `python make_csv.py -i images/ESP_037262_1845_RED_A_01_ORTHO.JP2 -ws 19`
3. `python combine_csv.py -ws 19`
4. `python trainer.py -ws 19`


# License

Copyright 2023, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.


# Acknowledgments

The research described in this publication was carried out in part at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration. This research has made use of the High Resolution Imaging Science Experiment on the Mars Reconnaissance Orbiter, under contract with the National Aeronautics and Space Administration. We acknowledge funding support from the National Aeronautics and Space Administration (NASA) Mars Data Analysis Program (MDAP) Grant Number NNH19ZDA001N.

## Paper Abstract

We developed an automated rock detection algorithm using a random forest (Breiman, 2001) ensemble classifier consisting of 10 individual models trained through k-fold cross-validation on approximately 4,000 manually labeled boulders in HiRISE imagery. The ensemble approach improves robustness and accuracy by combining multiple independent predictions, reducing overfitting and handling noise in the input data more effectively than single classifiers. Each model processes 11x11 pixel windows of standardized pixel values which limits the maximum detectable boulder size to approximately 3-6m depending on image resolution. To reduce false positives and influence from bad pixels we limit a minimum detection size of 3 pixels corresponding to ~1-1.5 meter. The standardization of pixel values acts as an automated contrast adjustment, eliminating the need for manual tuning and improving performance over previous techniques (Banfield 2011, Golembeck 2012, 2021). We employ a sliding window approach across the full image to generate a complete segmentation map for all boulders, which is then integrated with brain coral terrain masks generated through convolutional neural networks (Pearson et al., 2024) to enable comparative analysis of rock distributions.

After generating the initial segmentation map, we isolate individual boulders and analyze their shadow characteristics along with local elevation context. The extent of each boulder's shadow is measured and combined with the solar incidence angle from the image metadata to compute approximate boulder dimensions. To analyze the relationship between rock placement and local topography, we compute relative elevation by fitting and subtracting a plane from 100x100 m² regions surrounding each detected boulder. This local detrending removes bias from regional slopes and allows us to focus on boulder positions relative to immediate topographic features like ridges and depressions. Our analysis is performed using orthorectified images and HiRISE-derived digital elevation models (DEMs) to ensure precise spatial alignment. The combination of ensemble classification, automated shadow analysis, terrain masks, and detrended topographic data allows us to quantify rock densities and local elevation distributions across brain coral terrain systematically. We analyzed 10 different images measuring rock concentrations and elevation distributions both within brain coral terrain and in adjacent control regions outside the terrain masks, enabling robust testing of competing formation hypotheses through direct comparison of rock populations inside and outside these distinctive landforms

