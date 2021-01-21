# DIGEiZ Technical Challenge
Remove corrupted frames and find the right frame order.

## Algorithm

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