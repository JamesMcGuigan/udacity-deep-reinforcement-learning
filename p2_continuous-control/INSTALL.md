
# Download Unity Environment

These versions of the Unity Environment don't run on Ubuntu 20.04 
```
# wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
# unzip Reacher_Linux.zip
# mv Reacher_Linux Reacher_Linux_One
# 
# wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
# unzip Reacher_Linux.zip
# mv Reacher_Linux Reacher_Linux_Twenty
# 
# wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip
# unzip Crawler_Linux.zip 
```

Unsuccessful fixes:
```
export PATH="~/.anaconda3/envs/drlnd/bin/:$PATH"
sudo sysctl dev.i915.perf_stream_paranoid=0
find ./ -name '*.x86*' | xargs -L1 chmod a+x -v

pip3 install -r requirements.txt
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module
apt-get install lib32stdc++6
```

These github versions of Unity Environment seem to work better. Unsure of original source:
```
git clone https://github.com/ainvyu/p2-continuous-control.git
cp -r p2-continuous-control/unity/Reacher_Linux_NoVis/      ./
cp -r p2-continuous-control/unity/Reacher_One_Linux_NoVis/  ./
rm -rvf p2-continuous-control

git clone https://github.com/chihoxtra/continuous_actions_rl/
cp -r continuous_actions_rl/unity/Crawler_Linux_NoVis/ ./
rm -rvf continuous_actions_rl

git subtree add --prefix p2_continuous-control/DeepRL https://github.com/ShangtongZhang/DeepRL master --squash
```

Then install drlnd conda environment and python dependencies
```
cd p2_continuous-control
conda create --name drlnd python=3.6 
conda activate drlnd
python -m ipykernel install --user --name drlnd --display-name "drlnd"
pip install -r requirements.txt
```
