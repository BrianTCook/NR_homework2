#!/bin/bash

echo ""
echo "-------------------------------------"
echo "Brian Cook's homework 2 solutions"
echo "-------------------------------------"

echo ""
echo "-------------------------------------"
echo "Running the Python scripts for each part"
echo "-------------------------------------"

echo ""
echo "-------------------------------------"
echo "Problem 1, Part A"
echo "-------------------------------------"

python3 homework2problem1parta.py


echo ""
echo "-------------------------------------"
echo "Problem 1, Part B"
echo "-------------------------------------"

python3 homework2problem1partb.py

echo ""
echo "-------------------------------------"
echo "Problem 1, Part C"
echo "-------------------------------------"

python3 homework2problem1partc.py

echo ""
echo "-------------------------------------"
echo "Problem 1, Part D"
echo "-------------------------------------"

python3 homework2problem1partd.py

echo ""
echo "-------------------------------------"
echo "Problem 1, Part E"
echo "-------------------------------------"

wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
python3 homework2problem1parte.py

echo ""
echo "-------------------------------------"
echo "Problem 2"
echo "-------------------------------------"

python3 homework2problem2.py

echo ""
echo "-------------------------------------"
echo "Problem 3"
echo "-------------------------------------"

python3 homework2problem3.py

echo ""
echo "-------------------------------------"
echo "Problem 4, Part A"
echo "-------------------------------------"

python3 homework2problem4parta.py

echo ""
echo "-------------------------------------"
echo "Problem 4, Part B"
echo "-------------------------------------"

python3 homework2problem4partb.py

echo ""
echo "-------------------------------------"
echo "Problem 4, Part C"
echo "-------------------------------------"

python3 homework2problem4partc.py

#ffmpeg -framerate 25 -pattern_type glob -i "frame_*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 homework2problem4partc.mp4
#ffmpeg -r 30 -f image2 -s 64x64 -i 'frame_*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p homework2problem4partc.mp4

convert -delay 4 'frame_*.png' -loop 0 homework2problem4partc.gif

echo "Was having issues with the given ffmpeg command, used gif maker from my own research"

echo ""
echo "-------------------------------------"
echo "Problem 4, Part D"
echo "-------------------------------------"

#python3 homework2problem4partd.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part A"
echo "-------------------------------------"

python3 homework2problem5parta.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part B"
echo "-------------------------------------"

python3 homework2problem5partb.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part C"
echo "-------------------------------------"

echo ""
echo "Didn't have time to do 5c-g"
echo "" 

#python3 homework2problem5partc.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part D"
echo "-------------------------------------"

#python3 homework2problem5partd.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part E"
echo "-------------------------------------"

#python3 homework2problem5parte.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part F"
echo "-------------------------------------"

#python3 homework2problem5partf.py

echo ""
echo "-------------------------------------"
echo "Problem 5, Part G"
echo "-------------------------------------"

#python3 homework2problem5partg.py


echo ""
echo "-------------------------------------"
echo "Problem 6"
echo "-------------------------------------"

wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
python3 homework2problem6.py


echo ""
echo "-------------------------------------"
echo "Problem 7"
echo "-------------------------------------"

wget strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5
python3 homework2problem7.py

echo ""
echo "-------------------------------------"
echo "Problem 8"
echo "-------------------------------------"

echo ""
echo "Didn't have time to do 8"
echo "" 
#python3 homework2problem8.py

echo ""
echo "-------------------------------------"
echo "LaTeX'ing the report"
echo "-------------------------------------"

pdflatex homework2_cook.tex
bibtex homework2_cook.aux
pdflatex homework2_cook.tex
pdflatex homework2_cook.tex

echo ""
echo "-------------------------------------"
echo "My code/outputs are saved as homework2_cook.pdf"
echo "-------------------------------------"
