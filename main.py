import neat
import numpy as np
import os
import pickle
import pygame
from functools import cache
from PIL import Image
from scipy import ndimage
from skimage.transform import resize

"""" 
             ██████╗  ██████╗  ███╗   ██╗ ███████╗ ██╗  ██████╗  
            ██╔════╝ ██╔═══██╗ ████╗  ██║ ██╔════╝ ██║ ██╔════╝  
            ██║      ██║   ██║ ██╔██╗ ██║ █████╗   ██║ ██║  ███╗ 
            ██║      ██║   ██║ ██║╚██╗██║ ██╔══╝   ██║ ██║   ██║ 
            ╚██████╗ ╚██████╔╝ ██║ ╚████║ ██║      ██║ ╚██████╔╝ 
             ╚═════╝  ╚═════╝  ╚═╝  ╚═══╝ ╚═╝      ╚═╝  ╚═════╝  
"""
# Amount of generations.
amount_generations = 300

# Exit on first finish-line. This prevents an infinite loop where if the model is trained perfectly, it will keep running.
exit_finish_line = True

# Path to the track image. For making custom tracks, check the GitHub readme.
path_track = "Files/track-1.png"

# Path to the car image.
path_car = "Files/car.png"

# Path to the .pkl file in which the AI will be saved and loaded.
path_ai_file = "Files/winner-track-1.pkl"

# Replay the genome specified in path_ai_file.
replay_genome = False

# Frames Per Second if replay is animated.
# This is needed because otherwise this will go brrrrrrrrr.
fps = 144

# Path to the config-feedforward.txt file, AKA. the NEAT config.
# This is where you will find the Neural Network settings such as population size and biases.
path_feedforward = "Files/config-feedforward.txt"

# This will toggle the NEAT stats: fitness, id, size, adj fit, et cetera.
neat_statistics = True


""""
             ██████╗  ██████╗  ███████╗  ███████╗ 
            ██╔════╝ ██╔═══██╗ ██╔═══██╗ ██╔════╝ 
            ██║      ██║   ██║ ██║   ██║ █████╗   
            ██║      ██║   ██║ ██║   ██║ ██╔══╝   
            ╚██████╗ ╚██████╔╝ ██████╔═╝ ███████╗ 
             ╚═════╝  ╚═════╝  ╚═════╝   ╚══════╝ 
"""
pygame.init()
clock = pygame.time.Clock()
size = [600, 450]  # Size of the PyGame window
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Car Racing: NEAT')  # Name of the PyGame window

font = pygame.font.Font('freesansbold.ttf', 20)  # Font used for the text displayed
white = (255, 255, 255)

track = pygame.image.load(os.path.join(path_track))  # Load the track image
track_array = np.array(Image.open(os.path.join(path_track)).convert(colors=5))  # Process track image into Numpy array

best_fit = 0
generation = 0

# Lists that can be used for diagnostics
avg_performance_list = []
best_performance_list = []
genome_count_list = []

# Default variables for the car
car = pygame.image.load(os.path.join(path_car))  # Load car image
car_coords_x = round(np.average(np.where(track_array == [255, 0, 0])[1]))
car_coords_y = round(np.average(np.where(track_array == [255, 0, 0])[0]))
car_coords = (car_coords_x, car_coords_y)  # Get average x, y location of every red pixel. This is used as a finish- and start line
car_angle = 90
car_rect = car.get_rect()
car_rect.center = (car_coords[0] + car.get_width(), car_coords[1])

# Required for getting the location of the text. The non-rect variable doesn't have a .width property
# This is only required for the right-sided text, because the left text is fixed.
text_genome_generation = font.render(f"Generation: 0", True, white)
text_genome_generation_rect = text_genome_generation.get_rect()
text_genome_generation_rect.topleft = (size[0]-text_genome_generation_rect.width, 0)

text_genome_best = font.render(f"Best Fit: 0", True, white)
text_genome_best_rect = text_genome_best.get_rect()
text_genome_best_rect.topleft = (size[0]-text_genome_best_rect.width, 20)


def replay_genome_func():
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path_feedforward)

    # Unpickle saved winner
    with open(path_ai_file, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    main(genomes, config)


def run():
    """" This manages the NEAT stuff:
    - Makes a config
    - Gives output like statistics
    - Trains the genomes
    - Develops the genomes
    - Et cetera
    """

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(path_feedforward))

    if neat_statistics:
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main, amount_generations)

    # Save the winner object
    with open(path_ai_file, "wb") as f:
        pickle.dump(winner, f)

    print(best_performance_list)
    print(avg_performance_list)
    print(genome_count_list)


@cache
def rotate_array(angle: int, coords: tuple) -> np.array:
    """" Rotates a Numpy array, but needs coordinates to get the area.
    The ladder is due to limitations of @cache, which speeds up the code A LOT, so it's worth the trouble.

    :param angle: The angle of the car/genome.
    :param coords: The coordinates of the car/genome.
    """

    tenbyten_area = tuple([round(np.average(y)) for y in x] for x in track_array[coords[1] - 10:coords[1] + 10, coords[0] - 10:coords[0] + 10])
    arr = ndimage.rotate(tenbyten_area, angle, axes=(1, 0), mode='nearest')
    arr[arr <= 53] = 0
    arr[arr > 53] = 1
    arr = np.array([[int(str(x)[0]) for x in y] for y in resize(arr, (10, 10))])
    arr[arr <= 3] = 0
    arr[arr > 3] = 1
    return arr


@cache
def new_coordinates(coordinates: tuple, angle: int, speed: int) -> (int, int):
    """" Calculates the new coordinates based on the speed, angle and current.

    :param coordinates: The coordinates of the car/genome.
    :param angle: The angle of the car/genome.
    :param speed: The speed of all cars/genomes.
    """

    raw_coords_x = angle / 180 - 1
    if raw_coords_x > 0.5:
        raw_coords_x = 2 - angle/180
    elif -0.5 > raw_coords_x:
        raw_coords_x = -angle/180

    coords_x = speed * raw_coords_x * 2
    coords_y = speed - abs(coords_x)

    if not 270 > angle > 90:
        coords_y = -coords_y

    # This code might come in handy in the future. This calculates the difference between new and old coords.
    """if increasements:
        return round(coords_x), round(coords_y)
    else:"""
    return round(coordinates[0] + coords_x), round(coordinates[1] + coords_y)


def main(genomes: list, config):
    """" This manages everything from drawing the cars to the variables like genome fitness

    :param genomes: List with NEATs genomes.
    :param config: The NEAT config, basically Neural Network settings.
    """

    global best_fit, avg_performance_list, best_performance_list, generation

    net_array = {}
    coords_array = {}
    angle_array = {}

    best_fit_gen = 0
    speed = 5
    generation += 1

    for genome_id, genome in genomes:
        net_array[genome_id] = neat.nn.FeedForwardNetwork.create(genome, config)
        angle_array[genome_id] = car_angle
        coords_array[genome_id] = car_coords
        genomes[genomes.index((genome_id, genome))][1].fitness = 0

    while len(coords_array) > 0:
        screen.fill([255, 255, 255])
        screen.blit(track, (0, 0))
        text_genome_count = font.render(f"Genomes: {len(coords_array)}", True, white)  # Make a text
        text_genome_fitness = font.render(f"Avg Fit: {round(np.average([x[1].fitness for x in genomes]), 2)}", True, white)
        text_genome_gen_best = font.render(f"Best Fit Gen: {best_fit_gen}", True, white)

        text_genome_generation = font.render(f"Generation: {generation}", True, white)
        text_genome_generation_rect = text_genome_generation.get_rect()  # Get the rect of the text
        text_genome_generation_rect.topleft = (size[0]-text_genome_generation_rect.width, 0)  # Center the rect around the middle

        text_genome_best = font.render(f"Best Fit: {best_fit}", True, white)
        text_genome_best_rect = text_genome_best.get_rect()  # Get the rect of the text
        text_genome_best_rect.topleft = (size[0]-text_genome_best_rect.width, 20)  # Center the rect around the middle

        screen.blit(text_genome_count, (0, 0))
        screen.blit(text_genome_fitness, (0, 20))
        screen.blit(text_genome_gen_best, (0, 40))
        screen.blit(text_genome_generation, text_genome_generation_rect)
        screen.blit(text_genome_best, text_genome_best_rect)
        for genome_id, genome in genomes:
            if genome_id in coords_array:
                coords = coords_array[genome_id]
                angle = angle_array[genome_id]

                increasement_dir = round(net_array[genome_id].activate(rotate_array(angle, coords).flat)[0])

                if increasement_dir == 1:
                    angle += 10
                elif increasement_dir == -1:
                    angle -= 10

                if track_array[coords[1]][coords[0]].tolist() == [0, 127, 14]:
                    del coords_array[genome_id]
                    continue

                if exit_finish_line and track_array[coords[1]][coords[0]].tolist() == [255, 0, 0] and genome.fitness > 50:
                    if not replay_genome:
                        # Save the winner object
                        with open(path_ai_file, "wb") as f:
                            pickle.dump(genome, f)

                        print(best_performance_list)
                        print(avg_performance_list)
                        print(genome_count_list)

                        exit()
                    else:
                        replay_genome_func()


                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                if angle >= 360:
                    angle = 0
                elif angle < 0:
                    angle = 359

                genomes[genomes.index((genome_id, genome))][1].fitness += 1

                coords = new_coordinates(coords, angle, speed)
                coords_array[genome_id] = coords

                spoof_car = pygame.transform.rotate(car, angle - 90)
                spoof_car_rect = spoof_car.get_rect()
                spoof_car_rect.center = (coords[0], coords[1])
                angle_array[genome_id] = angle

                screen.blit(spoof_car, spoof_car_rect)

        for _, genome in genomes:
            if genome.fitness > best_fit:
                best_fit = genome.fitness

            if genome.fitness > best_fit_gen:
                best_fit_gen = genome.fitness

        pygame.display.flip()
        if replay_genome is True:
            clock.tick(fps / 2)
    best_performance_list.append(best_fit)
    avg_performance_list.append(round(np.average([x[1].fitness for x in genomes]), 2))
    genome_count_list.append(len(genomes))


if __name__ == '__main__':
    if replay_genome:
        replay_genome_func()
    else:
        run()
