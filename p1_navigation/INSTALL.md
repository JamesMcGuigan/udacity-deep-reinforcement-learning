# Install

Install Banana
```
cd p1_navigation
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
```

Install mujoco200
- https://www.roboti.us/index.html
- https://medium.com/@chinmayburgul/setting-up-mujoco200-on-linux-16-04-18-04-38e5a3524c85

```
cd p1_navigation
lynx https://www.roboti.us/index.html
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip ~/Downloads/mujoco200_linux.zip 
mkdir ~/.mujoco
mv ~/Downloads/mujoco200_linux.zip ~/.mujoco/
cd ~/.mujoco/
unzip mujoco200_linux.zip 
cd mujoco200_linux/
mv ~/Downloads/LICENSE.txt ~/.mujoco/
mv ~/Downloads/mjkey.txt   ~/.mujoco/
```

Install python dependencies
```
cd p1_navigation
conda create --name drlnd python=3.6 
conda activate drlnd
python -m ipykernel install --user --name drlnd --display-name "drlnd"
pip install -r requirements.txt
```
