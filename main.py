from typing import Any, Generator

import neat
import numpy as np
import pickle
import math
import pandas as pd
from functools import cache
from PIL import Image
from numba import jit

"""" 
             ██████╗  ██████╗  ███╗   ██╗ ███████╗ ██╗  ██████╗  
            ██╔════╝ ██╔═══██╗ ████╗  ██║ ██╔════╝ ██║ ██╔════╝  
            ██║      ██║   ██║ ██╔██╗ ██║ █████╗   ██║ ██║  ███╗ 
            ██║      ██║   ██║ ██║╚██╗██║ ██╔══╝   ██║ ██║   ██║ 
            ╚██████╗ ╚██████╔╝ ██║ ╚████║ ██║      ██║ ╚██████╔╝ 
             ╚═════╝  ╚═════╝  ╚═╝  ╚═══╝ ╚═╝      ╚═╝  ╚═════╝  
"""
# Path to the track image. For making custom tracks, check the GitHub readme.
path_track = "Files/track-1.png"

# Speed of the car in pixels
speed = 5

# Angle in degrees in which the car can turn in 1 move
# Intensity = 20 gives -20 to 20 degrees in turns.
angle_intensity = 20

# The start angle of the car
# You should change this accordingly when customizing the track
car_angle = 90

# Finish line color [R, G, B]
flc = [255, 0, 0]

# Road color [R, G, B]
rc = [64, 64, 64]

# Path to the car image.
path_car = "Files/car.png"

# Path to the .pkl file in which the AI will be saved and loaded.
path_ai_file = "Files/winner-track-1.pkl"

# Path to excel file containing the tracked data
path_output_stats = "Files/statistics-track-1.xlsx"

# Replay the genome specified in path_ai_file.
replay_genome = False

# Frames Per Second if replay is animated.
# This is needed because otherwise this will go brrrrrrrrr.
fps = 10

# Path to the config-feedforward.txt file, AKA. the NEAT config.
# This is where you will find the Neural Network settings such as population size and biases.
path_feedforward = "Files/config-feedforward.txt"

# Max amount of generations.
amount_generations = 5000

# Exit on first finish-line. This prevents an infinite loop where if the model is trained perfectly, it will keep running.
exit_finish_line = True

# This will toggle the NEAT stats both in the console
neat_statistics = True

# Toggle the pygame GUI
pygame_toggle = True

""""
             ██████╗  ██████╗  ███████╗  ███████╗ 
            ██╔════╝ ██╔═══██╗ ██╔═══██╗ ██╔════╝ 
            ██║      ██║   ██║ ██║   ██║ █████╗   
            ██║      ██║   ██║ ██║   ██║ ██╔══╝   
            ╚██████╗ ╚██████╔╝ ██████╔═╝ ███████╗ 
             ╚═════╝  ╚═════╝  ╚═════╝   ╚══════╝ 
"""
if __name__ == "__main__":
    track_array_visual = np.array(Image.open(path_track).convert(colors=5))  # Process track image into Numpy array

    track_array = np.array([[np.average(y) for y in x] for x in track_array_visual])
    start_car_coords = np.where(track_array == np.average(flc))
    car_coords_x = round(np.average(start_car_coords[1]))
    car_coords_y = round(np.average(start_car_coords[0]))
    track_array = np.vectorize(lambda x: 0 if x != np.average(rc) and x != np.average(flc) else 1)(track_array)
    car_coords = (car_coords_y, car_coords_x)

    # All integers
    tracked_dataframe = pd.DataFrame({'Generation': [], 'Average Fitness': [], 'Best Fitness (All Time)': [], 'Best Fitness (Generation)': [], 'Genome Count': []}).set_index("Generation")
    with open("Files/config-feedforward.txt") as f:
        threshold = [int(l.split("=")[1]) for l in f.readlines() if "fitness_threshold" in l][0]

    best_fit = 0
    generation = 0

    if pygame_toggle:
        import pygame

        pygame.init()
        clock = pygame.time.Clock()
        pygame.display.set_caption('Car Racing: NEAT')  # Name of the PyGame window

        font = pygame.font.Font('freesansbold.ttf', 20)  # Font used for the text displayed
        size = [len(track_array[0]), len(track_array)]  # Size of the PyGame window
        screen = pygame.display.set_mode(size)

        # Default variables for the car
        track = pygame.image.load(path_track).convert(screen)  # Load the track image
        car = pygame.image.load(path_car).convert(screen)  # Load car image
        car_rect = car.get_rect()
        car_rect.center = (car_coords[0] + car.get_width(), car_coords[1])


        # Required for getting the location of the text. The non-rect variable doesn't have a .width property
        # This is only required for the right-sided text, because the left text is fixed.
        text_genome_generation = font.render(f"Generation: 0", True, (255, 255, 255))
        text_genome_generation_rect = text_genome_generation.get_rect()
        text_genome_generation_rect.topleft = (size[0] - text_genome_generation_rect.width, 0)

        text_genome_best = font.render(f"Best Fit: 0", True, (255, 255, 255))
        text_genome_best_rect = text_genome_best.get_rect()
        text_genome_best_rect.topleft = (size[0] - text_genome_best_rect.width, 20)

    if neat_statistics and not replay_genome:
        from requests import Session
        from multiprocessing import Process
        from stats import neat_stats

        sess = Session()
        stats_process = Process(target=neat_stats)
        stats_process.start()


def replay_genome_func():
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path_feedforward)
    with open(path_ai_file, "rb") as f:
        genome = pickle.load(f)

    genomes = [(1, genome)]
    main(genomes, config)


def run():
    """" This manages the NEAT stuff:
    - Makes a config
    - Gives output like statistics
    - Trains the genomes
    - Develops the genomes
    - Et cetera
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path_feedforward)

    p = neat.Population(config)
    if neat_statistics:
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
    try:
        winner = p.run(main, amount_generations)
    except Exception as e:
        if neat_statistics:
            stats_process.terminate()
        raise e

    # Save the winner object
    with open(path_ai_file, "wb") as f:
        pickle.dump(winner, f)
    print("Hit threshold maximum.")
    if neat_statistics:
        stats_process.terminate()


@cache
@jit(nopython=True)
def diff(x: int, y: int) -> int:
    phi = abs(x-y) % 360
    return 360 - phi if phi > 180 else phi


@jit
def dist(c1: tuple[int, int], c2: tuple[int, int]) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


@jit
def map_area(angle: int, coords: tuple) -> list[float]:
    """" Basically gets the coordinates in a 180 degree circle in front of the car

    :param angle: The angle of the car/genome.
    :param coords: The coordinates of the car/genome.
    """

    return [dist(coords, get_collision(coords, (angle+x-180) % 360)) for x in range(-90, 91, 5)]


@jit(nopython=True)
def new_coordinates(coords: tuple, angle: int) -> (int, int):
    """" Calculates the new coordinates based on the speed, angle and current coordiantes.

    :param coords: The coordinates of the car/genome.
    :param angle: The angle of the car/genome.
    :param car_speed: The speed of the car/genome.

    """

    raw_coords_x = angle / 180 - 1
    if raw_coords_x > 0.5:
        raw_coords_x = 2 - angle / 180
    elif -0.5 > raw_coords_x:
        raw_coords_x = -angle / 180

    coords_x = speed * raw_coords_x * 2
    coords_y = speed - abs(coords_x)
    if not 270 > angle > 90:
        coords_y = -coords_y

    return int(coords[0] + coords_y), int(coords[1] + coords_x)


@jit
def get_collision(coords: tuple, angle: int) -> (int, int):
    """ Calculates the coordinates where the car will collide with the track using sin and cos

    :param coords: The coordinates of the car/genome.
    :param angle: The angle of the car/genome.
    """

    i = 0
    sin = math.sin(math.radians(angle))
    cos = math.cos(math.radians(angle))
    while i < max(track_array.shape):
        y, x = int(coords[0] + cos * i), int(coords[1] + sin * i)
        if track_array[y, x] == 0:
            return y, x
        i += 1
    return y, x


def main(genomes: list, config):
    """" This manages everything from drawing the cars to the variables like genome fitness

    :param genomes: List with NEATs genomes.
    :param config: The NEAT config, basically Neural Network settings.
    """

    global best_fit, generation, tracked_dataframe

    net_array = {}
    coords_array = {}
    angle_array = {}
    angle_history = {}
    #speed_array = {}
    best_fit_gen = 0
    generation += 1

    start_angle_history = (car_angle, car_angle, car_angle, car_angle, car_angle, car_angle, car_angle)
    for genome_id, genome in genomes:
        net_array[genome_id] = neat.nn.FeedForwardNetwork.create(genome, config)
        angle_array[genome_id] = car_angle
        angle_history[genome_id] = start_angle_history
        coords_array[genome_id] = car_coords
        #speed_array[genome_id] = speed
        genomes[genomes.index((genome_id, genome))][1].fitness = 0

    while len(coords_array):
        if pygame_toggle:
            screen.blit(track, (0, 0))
            if neat_statistics:
                text_genome_count = font.render(f"Genomes: {len(coords_array)}", True, (255, 255, 255))  # Make a text
                text_genome_fitness = font.render(f"Avg Fit: {round(np.average([x[1].fitness for x in genomes]), 2)}", True, (255, 255, 255))
                text_genome_gen_best = font.render(f"Best Fit Gen: {round(best_fit_gen, 2)}", True, (255, 255, 255))

                text_genome_generation = font.render(f"Generation: {generation}", True, (255, 255, 255))
                text_genome_generation_rect = text_genome_generation.get_rect()  # Get the rect of the text
                text_genome_generation_rect.topleft = (size[0] - text_genome_generation_rect.width, 0)  # Center the rect around the middle

                text_genome_best = font.render(f"Best Fit: {round(best_fit, 2)}", True, (255, 255, 255))
                text_genome_best_rect = text_genome_best.get_rect()  # Get the rect of the text
                text_genome_best_rect.topleft = (size[0] - text_genome_best_rect.width, 20)  # Center the rect around the middle

        for genome_id, genome in genomes:
            if genome_id in coords_array:
                coords = coords_array[genome_id]
                angle = angle_array[genome_id]
                if track_array[coords] == 0 or diff(angle_history[genome_id][0], angle) > 90:
                    del coords_array[genome_id]
                    continue

                if exit_finish_line and track_array_visual[coords].tolist() == flc and genome.fitness > 20:
                    if not replay_genome:
                        with open(path_ai_file, "wb") as f:
                            pickle.dump(genome, f)

                        print("Reached the finish line.")
                        if neat_statistics:
                            stats_process.terminate()
                        exit()
                    else:
                        replay_genome_func()
                elif threshold < genome.fitness:
                    return

                nn_output = net_array[genome_id].activate(map_area(angle, coords))
                #nn_output = net_array[genome_id].activate([speed_array[genome_id], *map_area(angle, coords)])
                #print(nn_output[1])
                #speed_array[genome_id] = max(5, speed_array[genome_id]+nn_output[1])

                angle = (angle + nn_output[0] * angle_intensity) % 360
                coords_array[genome_id] = new_coordinates(coords, angle)
                angle_array[genome_id] = angle
                angle_history[genome_id] = (*angle_history[genome_id][1:], angle)
                genomes[genomes.index((genome_id, genome))][1].fitness += 1

                if pygame_toggle:
                    spoof_car = pygame.transform.rotate(car, angle - 90)
                    spoof_car_rect = spoof_car.get_rect()
                    spoof_car_rect.center = coords_array[genome_id][::-1]
                    screen.blit(spoof_car, spoof_car_rect)

                    if neat_statistics:
                        for x in range(-90, 91, 15):
                            coords_translated = get_collision(coords, angle-180+x)
                            color = (255, 255, 255) if track_array_visual[coords_translated].tolist() == list(rc) else (255, 0, 0)
                            pygame.draw.line(screen, color, coords_translated[::-1], coords[::-1], 1)

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            exit()
        bfg = max(genomes, key=lambda x: x[1].fitness)[1]
        best_fit = max(bfg.fitness, best_fit)
        best_fit_gen = bfg.fitness

        if not replay_genome:
            with open(path_ai_file + "-unfinished", "wb") as f:
                pickle.dump(max(genomes, key=lambda x: x[1].fitness)[1], f)

        if pygame_toggle:
            if neat_statistics and not replay_genome:
                screen.blit(text_genome_count, (0, 0))
                screen.blit(text_genome_fitness, (0, 20))
                screen.blit(text_genome_gen_best, (0, 40))
                screen.blit(text_genome_generation, text_genome_generation_rect)
                screen.blit(text_genome_best, text_genome_best_rect)

            pygame.display.flip()
            if replay_genome:
                clock.tick(fps / 2)

    average_fitness = round(np.average([x[1].fitness for x in genomes]), 2)
    adf = pd.DataFrame({'Generation': [generation], 'Average Fitness': [average_fitness], 'Best Fitness (All Time)': [best_fit], 'Best Fitness (Generation)': [best_fit_gen], 'Genome Count': [len(genomes)]}).set_index('Generation')
    tracked_dataframe = tracked_dataframe.append(adf)
    tracked_dataframe.to_excel(path_output_stats, index=["Generation"])
    if neat_statistics:
        try:
            sess.post("http://127.0.0.1:1234", json={"bf": best_fit, "af": average_fitness, "gc": len(genomes), "gbf": best_fit_gen, "gen": generation}, timeout=0.1)
        except Exception:
            print("Successfully sent data to stats server")


if __name__ == '__main__':
    if replay_genome:
        replay_genome_func()
    else:
        run()
