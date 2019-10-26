#!/bin/sh

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

fileid=1KWBRCNPnHn_qM9cu2LxrUielV9Ds0Pbq
filename=surveys.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf surveys.tar.gz
rm -f surveys.tar.gz
