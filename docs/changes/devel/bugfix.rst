Restore the supported base NumPy/SciPy dependency floors and make the ``viz``
extra install Seaborn alongside Matplotlib. SSP-SIR now uses the reference
default ``M = rank(data) - artifact rank`` exactly and errors when no positive
reconstruction rank remains.
