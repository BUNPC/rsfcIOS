# Description:
This GUI is used to get resting state functional connectivity maps and metrics for the optical intrensic data in the mouse brain.

# Input: 
Preprocessed optical intrensic data <br/>
A file name is provided for a MAT file that contains the preprocessed optical intrensic data with the following fields: <br/>
I [MxNx1xnT] -This is preprocessed optical intensity data where MxN is size of the image and nT is number of time points. </br>
I0 [MxN] - This is single image that is used to select the brain region and seed points for connectivity analysis. MxN is size of the image </br>
I0_unsampled [MxN] - This is single unsampled image. MxN is size of the unsampled image. </br>
tRS [1xnT] - This contains time points where the images were collected. nT is number of time points. </br>

# Instructions to run rsfcIOS GUI:
* Makesure rsfcIOS in the MATLAB path using the matlab command setpaths(path, genpath(cd)) from rsfcIOS GUI folder.
* Change directory to the folder containing your data files.
* Type rsfcIOS in the MATLAB command window (Do not open by double clicking angioReg.fig. It will open the GUI but it may not function properly).
* To load the data go to **File > Load Data**
* Radio button **Brain MASK ON** will select the brain region if the data folder contains the brain region file called rsfc_brainMask.mat.
* If you don't have selected brain region already then use **select brain** button to select the brain regions where you want to analyse the data.
* This option will allow you to multiple regions on the brain. Once you select the region, it will be saved as rsfc_brainMask in the current workin directory.
* After selecting the brain region press **Pre-process** button to process the data. 
* preprocessing step does the global signal regression and calculates interhemisphiric connectivity map.
* In some cases, like for stroke studies, you may need to multiple signal regression instead of global signal regression. So before doing the preprocess please select one of the radio button from **Signal regression options** panel. If you choose MSR(multiple signal regression), please select infarct region before doing preprocessing.
* During the preprocess it will ask you select bregma and lambda on the brain image. Please select bregma and lamdba points on top right corner image. This will be help to get interhemisphiric connectivity map.
* To do seed based analysis, press select seed button and then select a seed on the brain to get connectivty map with respective to that seed. If you want to save this map, please select **Save Image** radio button, it will pop open the seed connectivity map after you select the seed. You can save that image. 
