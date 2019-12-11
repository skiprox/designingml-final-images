#!/bin/bash

for i in images/*.jpg;
do convert $i -resize "128x128>" resized/"${i////_}";
done;
