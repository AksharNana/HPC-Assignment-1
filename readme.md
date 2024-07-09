
## Image Convolution
To build:
```
cd ImageConv
make
```
The input ``file.pgm`` must be in data folder.
The output is saved to the output folder.

The program parameters are:
KernalChoice InputFileName.pgm OutputFileName.pgm factor (in this order)

Available kernals:
0: Sharpen (3x3) 
1: Emboss (3x3)
2: Emboss (5x5)
3: Average (5x5)

The factor argument: If you want to multiply the kernel value by a factor, set it here (otherwise set it to 1)

E.g.: Use average kernel, factor = 0.04
ImageConv 3 InputFileName.pgm OutputFileName.pgm 0.04

## K-Nearest
To build:
```
cd KNearest
```

Three make options available:

Serial Implementation:
Run ``make`` or ``make serial``

Parallel Task Implementation:
Run ``make task``

Parallel Section Implementation:
Run ``make section``

The program takes arguments: Nuber_ref_points dimension
e.g. Run on 10000 reference points in 128 dim.
./Knearest 10000 128