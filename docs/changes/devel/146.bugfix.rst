Fixed segmented ``BSSCCA`` switching abruptly between overlapping blocks: each block
used to take full weight at its first sample, so the output mix jumped from one block to
an even mix of two. Neighbouring blocks now cross-fade in the middle of their overlap.
Fixed ``BSSCCA`` fitting and modifying channels marked bad in MNE input; they are now
left out of the decomposition and pass through unchanged.
