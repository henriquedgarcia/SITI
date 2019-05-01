
# SITI
Python script to calculate SI/TI according recomendation ITU-T P.910.

This script was based on a [SITI](https://github.com/Telecommunication-Telemedia-Assessment/SITI) script from Telecommunication-Telemedia-Assessmentthat Github Group

**Depends on:**
- Numpy
- Pandas
- SkVideo
- Scipy
- Matplotlib (only to debug 

**Features:**
- Calculations use float32 arrays and output is rounded to 4 decimal places.
- Uses ffmpeg as backend (Supports most encodings).
- Saves output as CSV.
- Calculations are done considering only the luminance, as recommended by the ITU-T.
- Suport ffmpeg parameters
- Ability to be called by the command line or imported by another script.



