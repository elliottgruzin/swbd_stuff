for file in `cat swb_refined.txt`; do
  echo $file
  sox swb1/$file.sph -b 16 stereo_wavs/$file.wav
done
