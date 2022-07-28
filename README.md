# Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration

![banner](https://durrlab.github.io/C3VD/assets/img/sample.gif)

This repository contains the registration and rendering code used in *Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration*. Visit the [project webpage](https://durrlab.github.io/C3VD/) to learn more about this work.

## Prerequisites
### Software
* Ubuntu 20.04
* CMake>=3.20
* Nvidia Device Drivers>=450
* Nvidia CUDA>=11.1
* Nvidia Optix SDK 7.1.0
* OWL (*included as submodule*)
* OpenCV (*included as submodule*)
* Libcmaes (*included as submodule*)
* Eigen (*included as submodule*)
### Hardware
* NVIDIA GPU of Compute Capability 5.0 (Maxwell) or higher.

## Build Intructions
1. Pull the repository and associated submodules:
```
git clone 
git submodule init
```
2. Download the [Nvidia Optix SDK](https://developer.nvidia.com/designworks/optix/downloads/legacy) from the Nvidia Developer Portal and place it in the root directory. Rename the downloaded folder to "optix". 
3. Build the submodules and main executables:
```
mkdir build
cd build
cmake ..
make -j8
```

## Usage
The build will compile three executable files placed in the *bin* folder:
- *initialize*: launch a Graphical User Interface (GUI) for manually initializing the 3D model position
- *register*: optimize the model position that minimizes the alignment cost function
- *render*: render and save ground truth depth, surface normals, optical flow, and occlusion frames for every frame in the video sequence. Also output a coverage map for the entire video sequence. 

Before running any of the programs, create a new working directory for each video sequence, organized as follows:
```bash
.
└── VIDEO_DIR/        # working directory for the given video sequences
    ├── calib.txt     # omnidirectional camera calibration parameter file; formatted as width, height, cx, cy, a0, a2, a3, a4, c, d, e
    ├── config.ini    # parameter file
    ├── model.obj     # ground truth 3D model
    ├── pose.txt      # robot pose log; one pose per line, formatted <time in seconds> <homogenous pose in column-major form>
    ├── rgb/          # rgb image folder
    │   ├── 0.png         
    │   ├── 1.png
    │   │   ...
    │   └── N-1.png
    ├── depth/        # GAN-predicted depth image folder
    │   ├── 0.png         
    │   ├── 1.png
    │   │   ...
    │   └── N-1.png
    ├── results/      # registration results folder
    └── render/       # gt rendering output folder
```
A configuration file named *config.ini* should be placed in the video directory with the following variables defined:
- *X*: Handeye calibration matrix  (homogenous, column-major)
- *Ac*: Robot pose retained from the handeye calibration (homogenous, column-major) 
- *Bc*: Camera pose retained from the handeye calibration (homogenous, column-major)
- *poseOffset*: Temporal offset (in seconds) to synchronize the pose log with the video sequence. Frame 0 will then be paired with the pose at time *poseOffset* in the pose log.
- *T_init*: initial model transform in radians/millimeters (Rx,Ry,Rz,Tx,Ty,Tz) 
- *K*: number of target frames to sample from the video sequence for registration

### Manual Initialization
The *initialization* program launches a Graphical User Interface (GUI) that allows the user to manually perturb the model position to roughly align it with the video sequence. The model is initialized at *T_init* in the configuration file. Video frames are overlayed with renderings of the 3D model, and the camera pose is updated as the video is navigated. To run the program:

```
./render <VIDEO_DIR>
```

### Registration
In addition to the general parameters listed above, the following parameters should be added to the configuration file to run the registration program:
- *deltaR*: +/- parameter space bounds for rotation components of model position (radians)
- *deltaT*: +/- parameter space bounds for translation components of model position (millimeters)
- *popSize*: Population size for CMAES optimization
- *sigma*: Search sigma for CMAES optimization

To run the registration program:
```
./register <VIDEO_DIR>
```

Once the optimization is complete, the optimized pose is printed to the terminal window. Initial and final alignment images are saved in the *results* subdirectory.

### Ground Truth Rendering

Update the T_initial model position in the configuration file to the position result from the registration program. Then, run the ground truth rendering program:
```
./render <VIDEO_DIR>
```
Rendered ground truth files are saved in the *render* folder.

## Sample Video Sequence
A sample raw video sequence from the dataset is available for download [HERE](). Once uncompressed, the folder is ready to be run by the programs.  

## Reference
If you find our work useful in your research please consider citing our paper:
```
@article{bobrow2022,
  title = {Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration},
  author = {Taylor L. Bobrow, Mayank Golhar, Rohan Vijayan, Venkata S. Akshintala, Juan R. Garcia, and Nicholas J. Durr},
  journal = {arXiv:2206.08903},
  year = {2022},
}
```

## License
This work is licensed under CC BY-NC-SA 4.0
