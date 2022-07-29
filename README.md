# NEAT Car Racing
NeuroEvolution of Augmenting Topologies meets Car Racing.

This application basically trains cars to ride on a custom made track (more details in this readme!) using NEAT.

Got any suggestions (especially regarding the NEAT config)? Feel free to make an issue on the repo!

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

You must define the color values of the finish line, road and not-road in main.py.

Note the following things:
- The track must be 600x450px, or you should adjust the pygame window's size. 
- Only 650x450px tracks have been tested.
- Leave at least 15 pixels between different road parts, as the AI might get confused because of the input range.
