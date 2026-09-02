BSS-CCA
=======

This example uses a real EEG substrate with controlled broadband muscle-like
contamination to illustrate reference-free BSS-CCA. The method separates
components according to lagged temporal correlation, so broadband
low-correlation components can be attenuated without a dedicated reference
channel.

The controlled construction makes it possible to evaluate both artifact
recovery and change outside the contaminated periods. Low lagged correlation
is a selection heuristic, not an automatic artifact label.
