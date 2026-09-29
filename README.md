# SAM2CAD

## This small project was made mainly for speeding up my cad modeling process
### Works hella good for cookie cutters and custom organization/storage solutions

#### More features in the future!!!

For installing it you first need to have python 3.12 as minimum.

A GPU is highly recommended.

I personally use a Blackwell gpu (CU132 on pytorch). Feel free to experiment with different gpus and nvidia toolkit versions.

For installing the nvidia cuda toolkit just go to the official website

https://developer.nvidia.com/cuda/toolkit

For everything else do this steps

First: Create a Virtual environment with your python version

python3.12 -m venv venv

(If you're on windows do this for activating the venv: venv/scripts/activate)
(If you're on linux/mac do this for activating the venv: source venv/bin/activate)

Second: Install libraries

I personally used claude a lot for this, so there is some slop expected. Check out the requirements.txt file

For the libraries you should install pytorch first from here (if you already installed nvidia cuda toolkit do an nvidia-smi. If you dont. Whatchu waiting for? Again. Go to the link below and install that toolkit)

https://pytorch.org/get-started/locally/

Then just nuke a 

pip install -r requirements.txt

After those long downloads, you should be ready to go. 

Just run this command for segmenting the images

python app.py [dont write the square brackets nor whats inside. Here you  put the path to the image. I strongly recommend you to change the name to something easy to recognize, like img1]

The command should look something like

python app.py img1.jpeg

After you segmented the target image run this command for exporting to dxf:

python regions.py [[path_to_your_processed_images_folder]\[name_of_your_regions.json]] --longest-mm [size_on_mm]

the command should look something like this:

python regions.py processed\img1_regions.json --longest-mm 70