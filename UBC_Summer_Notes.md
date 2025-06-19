# UBC Summer Notes

## June 17th, 2025

- Shuming: 
  - Use data's oscillation over the zero line in the residue graph compared to a interpolated data to find the mixed layers 
- Classes structure to build:
  - number of each type of layers
  - length of mixed layer, connecting layer and interface in height 
  - length of interface in temperature 
- For interface: require minimum separate in temperature axis (done)
- In continuity(), add the condition for minimum number between 1s, or 2s to call it a valid layers (done)

## June 28th, 2025

- Class structure for layers:
  - Read from mask: starting and ending index of a layer 
  - Label the layers 
  - Then record other staff like length in depth and temperature 
- Global variable for each profile (in create_netcdf.py): number of sc, ml, int
