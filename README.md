# Colonoscopy 3D video dataset with paired depth from 2D-3D registration

![banner](https://durrlab.github.io/C3VD/assets/img/sample.gif)

This repository contains the registration and rendering code used in *Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration*. Visit the [project webpage](https://durrlab.github.io/C3VD/) to learn more about this work.

## Prerequisites
### Software
* Ubuntu 20.04
* CMake>=3.22
* Nvidia Device Drivers>=450
* Nvidia CUDA>=11.1
### Hardware
* NVIDIA GPU of Compute Capability 5.0 (Maxwell) or higher.

## Build Instructions
1. Install required third-party libraries:
```sudo apt install -y freeglut3-dev libglew-dev```

2. Pull the repository and associated submodules:
```
git clone 
cd C3VD
git submodule init
git submodule update
```
3. Build the submodules and main executables:
```
mkdir build
cd build
cmake ..
make -j8
```

## Usage
The build will compile three executable files placed in the *bin* folder:
- *align*: launches a Graphical User Interface (GUI) for manually initializing the 3D model position
- *register*: registers the 3D model to the target depth frames
- *rendergt*: renders and saves ground truth depth, surface normals, optical flow, and occlusion frames for every frame in the video sequence. Additionally, outputs poses for every frame and a coverage map for the entire video sequence. 

Before running any of the programs, create a new working directory for each video sequence, organized as follows:
```bash
.
└── SAMPLE_DIR/       # working directory for the given video sequences
    ├── config.ini    # parameter file
    ├── model.obj     # ground truth 3D model
    ├── model.mtl     # ground truth 3D model material
    ├── pose.txt      # robot pose log; one pose per line, formatted <time in seconds> <homogenous pose in column-major form>
    ├── mask.png      # binary corner mask for Olympus endoscopes
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
### Manual Initialization
The *align* program launches a Graphical User Interface (GUI) that allows users to manually perturb the model position to roughly align it with the video sequence. Video frames are overlayed with renderings of the 3D model, and the camera pose is updated as the video is navigated. The following parameters should be defined in the parameter file (config.ini):
- Omnidirectional camera intrinsics: *width*, *height*, *cx*, *cy*, *ao*, *a1*, *a2*, *a3*, *a4*, *c*, *d*, *e*
- *Acal*: Robot pose retained from the handeye calibration (homogenous, column-major) 
- *Bcal*: Camera pose retained from the handeye calibration (homogenous, column-major)
- *X*: Handeye calibration matrix  (homogenous, column-major)
- *modelTransform*: Initial model transform with 6 values: X-Y-Z axis rotation in radians and X-Y-Z translation in millimeters
- *poseStartTime*: Temporal offset (in seconds) to synchronize the pose log with the video sequence. Frame 0 is paired with pose at time *poseStartTime* in the pose log

To run the program:
```
./c3vd align <SAMPLE_DIR>
```
Parameters can be manipulated using inputs on the GUI window or using the keyboard. Press 'i' to print the keyboard input key to the terminal window.

### Registration
<COMING SOON>
After updating the <modelTransform></modelTransform> parameters in the configuration file with the model transform values from the alignment GUI, an optimization can be run to fine-tune the video alignment. In addition to the configuration parameters listed above, the following parameters should be added to the configuration file before running the registration program:
- *deltaR*: +/- parameter space bounds for rotation components of model position (radians)
- *deltaT*: +/- parameter space bounds for translation components of model position (millimeters)
- *popSize*: Population size for CMAES optimization
- *sigma*: Search sigma for CMAES optimization
- *K*: number of target frames to sample from the video sequence for registration

To run the program:
```
./c3vd register <SAMPLE_DIR>
```
Once the registration is complete, the optimized model transform is printed to the terminal window. Initial and final alignment images are saved in the *results* subdirectory.

### Ground Truth Rendering
Update the modelTransform parameter in the configuration file to the result from the registration program. Then, run the ground truth rendering program:
```
./c3vd rendergt <SAMPLE_DIR>
```
Rendered ground truth files are saved in the *render* folder.

## Sample Video Sequence
A sample raw video sequence from the dataset is available for download [HERE](https://drive.google.com/file/d/1Bmg0ATcwG-gFLAKB_dBKoAiY2suBx1_X/view?usp=sharing). Once uncompressed, the folder is ready to be run by the programs.  

## Reference
If you find our work useful in your research, please consider citing our paper:
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
