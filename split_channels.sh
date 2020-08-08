#!/bin/sh

mkdir ./dialogues_mono
for i in `cat swb_complete.txt`; do
	echo "VAR: $i"
	# leftChannelEnd="${i%.mix*}.g.wav"
	# leftChannel="./signals/dialogues_mono${leftChannelEnd#./signals/dialogues*}"
	#
	# rightChannelEnd="${i%.mix*}.f.wav"
	# rightChannel="./signals/dialogues_mono${rightChannelEnd#./signals/dialogues*}"
	# echo "LEFTCHANNEL: $leftChannel"
	sox swb1/$i.sph -b 16 -c 1 dialogues_mono/$i.g.wav remix 1
	sox swb1/$i.sph -b 16 -c 1 dialogues_mono/$i.f.wav remix 2

done
