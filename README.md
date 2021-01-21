# DIGEiZ Technical Challenge
Remove corrupted frames and find the right frame order.

## Algorithm

### Find all the corrupted frames

    They are removed due to histogram comparisons after converting the RGB values of the frame to a different space, to be less light dependant. 

    I tried all the different color spaces, and found out that for this set of frames, the XYZ color space gave the best results.

    As for the method of comparison, I selected after some tries the Correlation Comparison.

    In order to effectively find the corrupted frames, we want to find frames that are similar and frames that are unique.
    I implemented this by comparing each histogram with all the other histograms, and counting hom many of them have a histogram correlation distance (from 0.0 [no correlation] to 1.0 [same frame]) that is more than 0.8. This will then give me a percentage of how many frame are simal to this frame. If the percentage of frames similar to this frame is more than 50%, the frame is considered not corrupted.

### Find the correct flow
    aokdazokdaozkdzkdpazokdpoakd
    azpodapzodzapdkazpodk
    aopzkdpaozkdpozakdpazkdpoakpzdok
    aozkdpaokdpoazkdpoazkdpokpozakdpoazkd
    
## Requirements
- Python 3 (3.8)
    - numpy (1.19)
    - opencv-python (4.4)
    - matplotlib (3.3)
- C++ (C++11)
  - CMake (3.8)
  - OpenCV (4.4)


## How to run
Clone and get inside directory:
```bash
git clone https://github.com/givka/digeiz-corrupted.git
cd digeiz-corrupted
```

### Python version
I used this version to quickly change the different parameters and the logic of the algorithm.

```bash
python prototype.py
```
This will write `python_video.mp4` to the main directory.
### C++ version
```bash
mkdir build
cd build
cmake ..
make
./digeiz-corrupted
```
This will write `cpp_video.mp4` to the main directory.