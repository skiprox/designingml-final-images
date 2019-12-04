#!/bin/bash

for i in images/*.jpg;
do convert $i -resize "64x64>" resized/"${i////_}";
done;
