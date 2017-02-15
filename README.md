# WNV_model

West Nile Virus prediction model based on the 2015 Kaggle competition.

## Summary
The goal of this project is to better predict the presence of the West Nile Virus (WNV) in Chicago, which is carried by mosquitoes. Most people infected with WNV have no symptoms of illness and never become ill, but approximately one in five people will develop symptoms that can last for weeks or months. Less than 1% of those infected will develop more severe neurological symptoms that can lead to permanent impairment or even death. People over the age of 60 and those with chronic diseases are more at-risk for serious illness[<sup>1</sup>](https://www.cdc.gov/westnile/faq/genquestions.html).

This project takes elements from the winning models of the 2015 West Nile Virus Prediction [Kaggle competition](https://www.kaggle.com/c/predict-west-nile-virus). The final model creates a single set of predictions for each location to indicate the likelihood of having WNV.

This project is a joint effort within the City of Chicago between the Department of Innovation and Technology and the Department of Public Health.

## Dependencies
- python 3.X 
- pip 1.3 or greater
- wget (In case you don't have it, you can download it through [Homebrew](http://brew.sh/) on OSX or through [GnuWin](http://gnuwin32.sourceforge.net/packages/wget.htm) on Windows.)

The following Python packages are needed, and will be attempted to install during setup:

- numpy 1.7.1 or greater
- scipy 0.11 or greater
- pandas 
- [scikit-learn](http://scikit-learn.org/stable/install.html) (It is easy installing scikit-learn on OSX using [Anaconda](https://www.continuum.io/downloads))
- requests

## Step 0: Setting up a Python Virtual Environment

From the project root folder, execute at a command prompt: 

`./venv_setup.sh`

With the virtual environment in place, the environment can be invoked with 

- `source activate venv` on Linux/Mac.
- `source venv/Scripts/activate` on Windows (it may be necessary to modify the commands to add `.exe` for it to work).

This will activate a virtual environment (called "venv") where the user can install required packages that pertain to this project, without changing the system-wide Python installation.

Whenever you're ready to exit the Virtual Environment, type `source deactivate` on the terminal.

## Step 1: Installing virtual environment required packages

From the project root folder, and with the virtual environment activated, execute at a command prompt: 

`./venv_install_pkgs.sh`

This will attempt to install the dependencies required for the project. 

It is possible that some errors appear when attempting to install the libraries. You will need to satisfy dependencies for scikit-learn and pandas, which requires technical computing programs such as linear algebra libraries (BLAS, LAPACK, or ATLAS). It is suggested you roughly follow the instructions to install Theano (deep learning toolkit) at http://deeplearning.net/software/theano/install_ubuntu.html

Likely requirements (which can be installed system wide with apt-get, rpm, yum, or brew on OSX): `python3-numpy python3-scipy python3-dev python3-pip python3-nose g++ libopenblas-dev`. 

## Step 2: Downloading input data

Before downloading data, you should get a [token](http://www.ncdc.noaa.gov/cdo-web/token) from the NOAA website. After receiving it, store the string in the text file called `scripts/weather_noaa_token.txt`.

After the token has been saved, go to the project root folder and execute at a command prompt: 

`./download_input_data.sh`

Remember to activate the virtual environment when working on the project.

This script will first download:

- Mosquito trap data from the City of Chicago [Data Portal](https://data.cityofchicago.org/Health-Human-Services/West-Nile-Virus-WNV-Mosquito-Test-Results/jqe8-8r6s).
- Weather data from the National Oceanic and Atmospheric Administration (NOAA) [Climate Data Online API](http://www.ncdc.noaa.gov/cdo-web/webservices/v2). For information on variable names and their units, check `data/weather_readme.txt`.

Then it will merge both datasets to use on the predictive model.

## Step 3: Building model and getting predictions

As in the previous steps, from the project root folder and with the virtual environment activated, execute at a command prompt: 

`./build_model.sh <period_frequency_to_train_models>`

For example, this command will train a model with information from every 10 periods:

`./build_model.sh 10`

This step will get lengthier as the frequency of analysis is increased. If no arguments are specified, the models will be trained every 40 periods.

Predictions from the highest-rating model will be found on `data/predictions.csv`.

## Recap
Run the scripts in the following order:

	./venv_setup.sh	./venv_install_packages.sh	./download_input_data.sh	./build_model.sh <period_frequency_to_train_models>

Remember to have the dependencies installed, to save the NOAA API token in the specified file, and to use Python 3.X to run the code.

## Descriptions of folders
	WNV_model	
	├── README.md 				 
	├── LICENSE.md 				
	├── data 			# input/output files from input data downloads
	├── python 		# pythons scripts
	├── sandbox 		# files with exploratory analysis and preliminary results
	└── venv 			# created after venv is setup

## Known issues
virtualenv might have an OSError if Anaconda and Python 2.7.11 are installed. 
Be sure to use Python 3 to install it and activate it.

![i](https://cloud.githubusercontent.com/assets/16825698/16929927/9251d508-4cff-11e6-8485-32a2cbd06986.png)
