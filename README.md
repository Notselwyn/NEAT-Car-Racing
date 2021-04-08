# NEAT Car Racing
NeuroEvolution of Augmenting Topologies meets Car Racing.

This application basically trains cars to ride on a custom made track (more details in this readme!) using NEAT.

Got any suggestions (especially regarding the NEAT config)? Send me a DM at Discord: @Lau#1337

<img src="https://i.imgur.com/hi2z8OB.gif" width="400" height="300"></img>
## Dependencies

To install the following dependencies, run `pip install -r requirements.txt` in the source directory.

You can also manually install these packages running `pip install packagename` anywhere in your system. 

- neat-python
- numpy
- pygame
- scipy
- scikit-image

## Config

- amount_generations = 300
  - The amount of generations

- exit_finish_line = True
  - Exit on first finish-line. 
  - This prevents an infinite loop where if the model is trained perfectly, it will keep running.

- path_track = "Files/track-1.png"
  - Path to the track image. 
  - For making custom tracks, scroll down
 
- path_car = "Files/car.png"
  - Path to the car image.

- path_ai_file = "Files/winner-track-1.pkl"
  - Path to the .pkl file in which the AI will be saved and loaded.

- replay_genome = False
  - Replay the genome specified in path_ai_file.

- fps = 144
  - Frames Per Second if replay is animated.
  - This is needed because otherwise this will go brrrrrrrrr.

- path_feedforward = "Files/config-feedforward.txt"
  - Path to the config-feedforward.txt file, AKA. the NEAT config.
  - This is where you will find the Neural Network settings such as population size and biases.

- neat_statistics = True
  - This will toggle the NEAT stats: fitness, id, size, adj fit, et cetera.

## Making a custom track

You must follow these color codes strictly

- Road
  - #404040 
  - RGB(64, 64, 64)
- Grass (Kill zone)
  - #007F0E 
  - RGB(0, 127, 14)
- Finish- and start line
  - #FF0000 
  - RGB(255, 0, 0)

Note the following things:
- The track must be 600x450px, or you should adjust the pygame window's size. 
- Only 650x450px tracks have been tested.
- Leave at least 15 pixels between different road parts, as the AI might get confused because of the input range.

## How does it work (Steps)?
1. Pillow (a.k.a. PIL) turns the **track** image into a Numpy array called **track_array**. 


2. A bunch of PyGame mumbo jumbo happens like making a screen, printing cars, et cetera.


3. **run()** or **replay_genome_func()** gets called.

3,1. **run()** gets called.

3,1,1. NEAT generates **pop_size** amount of genomes and will decide wether or not there will be output containing statistics.

3,1,2. The training begins, thus **main()** gets called **amount_generations** times.


3,1,3. During these calls, NEAT will tweak the genomes' weights and biases based on the **genome.fitness** attribute. 

3,2. **replay_genome_func()** gets called.

3,2,1. A .pkl file containing the genome and specified as **path_ai_file** will get loaded.

3,2,2. **main()** gets called, and it will loop forever with the genome until the process is terminated (hit close window, or kill the process).

3,2,3. Keep in mind, if a genome is replayed, it will not be trained. 

4. **main()** gets called.

4,1. All genomes receive a default value of score, coordinates. They will also receive a custom "net".

4,2. The following will repeat until all genomes are dead:

4,2. Data about the genomes will be displayed in the window as text.

4,3. All living genomes will go through the following cycle:

4,4. Check if the genome is in the green area, if so it will die.

4,5. Get value a value 1, 0, -1, based on the net given in step 4,1. 

4,5,1. The input for the net are 100 nodes, gained in the following way:

4,5,2. **rotate_array()** gets called.

4,5,3. Based on the coordinates provided, it will work out a 20x20 area in **track_array**, with the car coords as a center.

4,5,4. For every pixel in this 20x20, get the average and put them into a numpy array.

4,5,5. **scipy.ndimage.rotate** rotates the array by **angle**%.

4,5,6. Every value below 53 becomes a 0, otherwise a 1. This is done for the following step.

4,5,7. Resize the array that can be any size, because it has been rotated, to 10x10.

4,5,8. Every value below 3 becomes a 0, otherwise a 1.

4,5,9. Return a 2D binary array. 

4,6. Increase angle by 10 if value is 1, don't adjust if it's 0, or decrease the angle by 10 if value is -1.

4,7. The new coordinates are calculated by the **new_coordinates** function, based on the distance, angle and current coordinates.

4,7,1. Work out angle / 180 - 1.

4,7,2. If the angle is above 270 or below 90, it will be going down instead of up. Basically 0.3 0.4 0.5 0.6 0.7 becomes 0.3 0.4 0.5 0.4 0.3.

4,7,3. Work out the new x based on **speed** * value_calculated_above * 2. Work out the new y based on **speed** - the_new_x. 

4,8. Draw the car.
