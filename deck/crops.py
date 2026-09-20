"""Cut the two constellation panels and the class-spectra grid out of the
committed figures for the deck. Run from the repository root:

    python deck/crops.py
"""
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent

im = Image.open(ROOT / "figures/01_constellations.png")
# Axes frames of the 16QAM and 64QAM panels (row 2, columns 1 and 2) in the
# 1960 x 1470 figure, with the panel title above and the tick labels around.
im.crop((70, 612, 462, 1012)).save(HERE / "c16qam.png")
im.crop((555, 612, 947, 1012)).save(HERE / "c64qam.png")

sp = Image.open(ROOT / "figures/08_radioml_class_spectra.png")
w, h = sp.size
# Drop the figure's own title band; the slide title does the talking.
sp.crop((0, int(h * 0.075), w, h)).save(HERE / "spectra24.png")
print("wrote c16qam.png c64qam.png spectra24.png")
