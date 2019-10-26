#!/bin/sh

# Warning: it takes a lot of space, be prepared (Yes, my teeth and ambitions are bared)

# Downloads a subset of the park slices.
while read -r line
do
  wget https://www.dropbox.com/sh/wrre8iii6wf2lyt/"$line"
  tarname="$(echo "$line" | cut -d '/' -f2)"
  echo "$tarname"
  tar -xvf "$tarname"
  rm "$tarname"
done < Extended-CMU-Seasons_links.txt
