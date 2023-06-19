while IFS= read -r dirName; do
    mkdir -p "$dirName"
done < newDirs.txt

