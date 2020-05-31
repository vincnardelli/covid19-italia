#nazionale
cd ../covid19-italia/nazionale
rm output/*
python3 SIR-R0.py
cp -R output/*.csv ../R/fit_modelli/

#macroregioni
cd ../macroregioni
rm output/*
python3 macroregioni.py
cp -R output/*.csv ../R/fit_modelli/

# regioni
cd ../regioni
rm export/*
python3 R0_map.py

# R
cd ../R
rm export/*
cp ../regioni/export/* export/
Rscript main.R
zip -r export/export2.zip export
aws s3 cp export/export2.zip s3://covstatit/
