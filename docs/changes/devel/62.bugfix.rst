Make documentation CI independent of external dataset hosts while keeping the
affected example pages available as source code. This also prevents a handled
dataset download failure from ending Sphinx early with a false-success status;
local documentation builds continue to execute the complete gallery by default.
