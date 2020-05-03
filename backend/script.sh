#nazionale
cd ../covid19-italia/nazionale
rm output/*
python3 SIR-R0.py
cp -R output/*.csv ../R/fit_modelli/

#macroregioni
cd ../macroregioni
rm output/*
python3 R0_regions_export.py
python3 SIR2_regions_export.py
cp -R output/*.csv ../R/fit_modelli/

# regioni
cd ../regioni
rm output/*
python3 R0_map.py
cp -R output/*.csv ../R/fit_modelli/
cp -R output/*.geojson ../R/fit_modelli/

# R
cd ../R
rm export/*
Rscript main.R
aws s3 cp export/ s3://vincnardelli.rdataexport/covstat/ --recursive
