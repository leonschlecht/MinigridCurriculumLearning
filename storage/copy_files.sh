#!/bin/bash

while read -r filename; do
    dirname=$(dirname "$filename")
    echo "$filename"
    cp "$filename" "~/Test12/MinigridCurriculumLearning/logs/$dirname/status.json"
done < filelist.txt

