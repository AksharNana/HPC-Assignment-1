sizes=(10000 100000 1000000)
dimension=(64 128 256)

echo ---------Starting Benchmark------------
echo =======================================
for i in "${dimension[@]}"
do
    echo "Testing Dimension: ${i}"
    echo =======================================
    echo Serial Version
    make
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            ./KNearest.exe ${size} ${i}
        echo ---------------------------------------
    done
    echo Parallel Tasks Version
    make task
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            ./KNearest.exe ${size} ${i}
        echo ---------------------------------------
    done
    echo Parallel Section Version
    make section
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            ./KNearest.exe ${size} ${i}
        echo ---------------------------------------
    done
    echo =======================================
done
echo --------------Clean Files--------------
make clean
echo ---------Benchmark Completed-----------