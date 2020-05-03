apt-get update
apt-get install git -y
git clone https://github.com/vincnardelli/covid19-italia.git

apt install -y awscli
#Python
apt install python3-pip -y
pip3 install numpy pandas geopandas scipy matplotlib xlrd jupyter
#R
add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
apt-get update
apt install r-base -y
apt-get install r-cran-dplyr -y
apt-get install libcurl4-openssl-dev
apt install libssl-dev libxml2-dev
apt install -y libudunits2-0 libudunits2-dev
apt install -y libgdal-dev
apt-get install -y libprotobuf-dev
apt-get install -y libjq-dev
apt-get install -y libv8-dev
apt-get install -y libprotobuf-dev protobuf-compiler
Rscript -e 'install.packages(c("dplyr", "writexl", "deSolve", "lubridate", "tidyr", "geojsonio", "ggplot2", "readr", "openxlsx", "stringr", "readxl))'
