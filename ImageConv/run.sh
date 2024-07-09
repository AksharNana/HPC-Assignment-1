echo ---------Starting Benchmark------------
make
echo =======================================
echo "Image 1"
    ./imageConv.exe 0 image21.pgm veg.pgm 1
echo "Image 2"
    ./imageConv.exe 1 man.pgm mans.pgm 1
echo "Image 3"
    ./imageConv.exe 2 mandrill.pgm mandrill_face.pgm 1
echo "Image 4"
    ./imageConv.exe 3 lena_bw.pgm lady.pgm 0.04
echo =======================================
echo --------------Clean Files--------------
make clean
echo ---------Benchmark Completed-----------