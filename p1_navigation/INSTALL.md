# Install

Install Banana
```
cd p1_navigation
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
unzip Banana_Linux.zip
```

Install mujoco200
- https://www.roboti.us/index.html
- https://medium.com/@chinmayburgul/setting-up-mujoco200-on-linux-16-04-18-04-38e5a3524c85

```
mkdir ~/.mujoco
cd    ~/.mujoco
lynx https://www.roboti.us/index.html
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://www.roboti.us/download/mujoco200_macos.zip
wget https://www.roboti.us/file/mjkey.txt
wget http://www.apache.org/licenses/LICENSE-2.0
unzip mujoco200_linux.zip
ln -sv mujoco200_linux mujoco200 
```

Create conda environment
```
cd p1_navigation
conda create --name drlnd python=3.6 
conda activate drlnd
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Install python dependencies
```
sudo pamac install libstdc++5
conda install libstdcxx-ng libstdcxx-devel_linux-64
conda update -n base -c defaults conda
conda install swig patchelf
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch===0.4.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install https://files.pythonhosted.org/packages/37/e5/e7504cb2ded511910c2a2e8f9c9e28af075850eb03a5c5a8daee5d7d9517/mujoco_py-2.1.2.14-py3-none-any.whl
pip install -r ../python/requirements.txt
pip install -r requirements.txt
```

