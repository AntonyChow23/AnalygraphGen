# AnalygraphGen
### A simple analygraph generator in cpp
## Initial setup
### 1.Install Xcode
### 2. Install Homebrew
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
### 3. Install OpenCV
```
brew install opencv
```
### 4. Install pkg-config
```
brew install pkg-config
```
### 5. check OpenCV path
##### check linker flags for OpenCV
```
    pkg-config --cflags --libs opencv
```
##### speficy location of opencv.pc if previous failed
```
    pkg-config --cflags --libs path/to/opencv.pc
```

## How to Run
### parameters
##### -l [following path for left image]
##### -r [following path for right image]
##### -i [following path for a combined image]
##### -T : generate true analygrph
##### -G : generate gray analygrph
##### -C : generate color analygrph
##### -H : generate half-color analygrph
##### -O : generate 3DTV-optimized analygrph
##### -D : generate DuBois analygrph
##### -R : generate Roscolux analygrph

```
make
./main -<function selection>  -i <path for a combined image> 
```
or
```
make
./main -<function selection> -l <path for left image> -r <path for right image>
```
An "analygraph.jpeg" file will be generated under the same directory of main




### Reference links:
https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003
https://stackoverflow.com/questions/23284473/fatal-error-eigen-dense-no-such-file-or-directory

Makefile:
https://www.codeproject.com/Questions/1275819/Makefile-for-Cplusplus-program-with-opencv
Eigen:
https://stackoverflow.com/questions/23284473/fatal-error-eigen-dense-no-such-file-or-directory

https://stackoverflow.com/questions/54971083/how-to-use-cvmat-and-eigenmatrix-correctly-opencv-eigen