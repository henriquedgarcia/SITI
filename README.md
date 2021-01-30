# SITI
Python script to calculate SI/TI similar to ITU-T P.910.

This script was based on a [SITI](https://github.com/Telecommunication-Telemedia-Assessment/SITI) script from Telecommunication-Telemedia-Assessmentthat Github Group

**Depends on:**
- Numpy
- Pandas
- SkVideo
- Scipy

**Features:**
- Calculations using median instead maximum value.
- Uses ffmpeg as backend (Supports most encodings and command line params).
- Saves output as CSV.
- Calculations are done considering only the luminance, as recommended by the ITU-T.
- Calcule some statistics about the SITI distribution along the video.
- Ability to be called by the command line or imported by another script.
