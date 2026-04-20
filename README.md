# Overview


## Code
The dependencies required for this code:
- Matplotlib
- Numpy
- Pillow
- Ipython
- Ipympl

Run the cells defining all the classes and functions

It is not recommended to run the Simulate() function as save or display the simulation is time consuming. Though if you wish to display, set the input parameter display = True, and you may want to remove the save_as input if you don’t want a gif saved.

The efficiency and accuracy cell at the bottom is what should be run if you want to quickly check that the method work. Ensure the Earth-Sun

## nbody.ipynb
Includes a System Class that handles the solving and animation of the system defined with a list of Mass class objects.

There are a variety of functions to build the galaxy system and three body system

Then there are a bunch of analysis cells that are using the integrators and simulate() function.

## alternative_nbody.ipynb

Used a different acceleration function
only plotted trajectory
