
# Download
```
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
unzip Reacher_Linux.zip
mv Reacher_Linux Reacher_Linux_One

wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
unzip Reacher_Linux.zip
mv Reacher_Linux Reacher_Linux_Twenty

wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip
unzip Crawler_Linux.zip
```

# System Fixes

Still getting 
```
export PATH="/home/jamie/.anaconda3/envs/drlnd/bin/:$PATH"
sudo sysctl dev.i915.perf_stream_paranoid=0
find ./ -name '*.x86*' | xargs -L1 chmod a+x -v

pip3 install -r requirements.txt
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
apt-get install lib32stdc++6
```
