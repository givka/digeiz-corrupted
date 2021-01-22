# DIGEiZ Technical Challenge
Remove corrupted frames and find the right frame order.

## Algorithm

### Find all the corrupted frames

They are removed due to histogram comparisons after converting the RGB values of the frame to a different space, to be less light dependant. 

I tried all the different color spaces, and found out that for this set of frames, the YUV color space gave the best results.

As for the method of comparison, I selected after some tries the Correlation Comparison.

In order to effectively find the corrupted frames, we want to find frames that are similar and frames that are unique.
I implemented this by comparing each histogram with all the other histograms, and counting hom many of them have a histogram correlation distance (from 0.0 [no correlation] to 1.0 [same frame]) that is more than 0.8. 

This will then give me a percentage of how many frame are similar to this frame. If the percentage of frames similar to this frame is more than 50%, the frame is not considered corrupted.

### Find the correct flow
I first considered using histogram comparison and a stack. It almost worked except for a few frames. This technique is not robust enough for complicated datasets.

I did Face detection, but actually, for this video, the face  moves slower than the camera, which makes tracking difficult, and not every video has faces on it.

I then used Features, Descriptors and Matchings to get the homography matrix between frames, which can give me the translation between frames, but it turns out it was costly and did not provide good result.

I finally used optical flow. At first, I select a random frame, and I compute the mean distance of optical flow vectors between this frame and all the other frames. The frame with the minimum distance is the closest.

I put these two frames in a stack and remove them from the frames to check.

I repeat this for the top of the stack and the bottom of the stack.
I check the minimum distance between the two and place the next frame in the stack accordingly (on the top if the minimum distance is from the top, or on the bottom)

I repeat this while there are still some frames to check.

The stack will be filled with the reordered frames.

For example:
```cpp
random_frame()                        ->  [0]
find_min_frame(0)                     ->  [1][0]
find_min_frame(1) | find_min_frame(0) ->  [2][1][0]
find_min_frame(2) | find_min_frame(0) ->  [2][1][0][3] 
find_min_frame(2) | find_min_frame(3) ->  [2][1][0][3][4]
```
After playing with the parameters, I manage to down sample the input images for the computation of the optical flow by 20, which resizes frames from (1920, 1080) to (96, 54). This value is specific for this video.

## Using another corrupted video
- Change the values of:
  - `INPUT_FILENAME`
  - `OUTPUT_FILENAME`

If the algorithm does not work on another video:
- Find all the corrupted frames:
  - change value of `MIN_HIST_CORREL = 0.8`  (`float[0,1]`)
  - change value of `MIN_HIST_SIMILAR = 0.5` (`float[0,1]`)
- Find the correct flow:
  - change value of `DOWNSAMPLE = 20` (`int[1,50]`)
    > Setting this to `1` will disable rescaling.
  - change value of `REVERSED = false` (`bool`)
    > The algorithm can not determine if the firsts two frames that will be added are in the right order. This is why this parameter is needed.

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

### Video format
If there is an error while writing the ordered frames video file, it may come from:
```py
filename = "python_video.mp4"
fourcc = cv.VideoWriter_fourcc(*'MJPG')
```
```cpp
auto filename = "../cpp_video.mp4";
auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
```
Change the extension and the fourcc to something that works on your computer.

## Ideas
- pass global variable values by argument
- instead of using a stack to store top and bot frames, store them in two seperate vectors and concat them at the end