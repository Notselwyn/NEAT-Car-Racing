from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import matplotlib.pyplot as plt

global avg_performance_list
global best_performance_list
global genome_count_list
global gen_performance_list
global manager


class Graph:
    def __init__(self, title, ylabel):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [])
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid()
        self.update([], 0)


    def update(self, arr, gen):
        self.lines.set_xdata(range(int(gen)))
        self.lines.set_ydata(arr)
        if len(arr) != 0:
            self.ax.set_xlim(0, int(gen))
            self.ax.set_ylim(0, int(max(arr)*1.10))
        self.ax.relim()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class Manager:
    def __init__(self, labels: list[tuple]):
        plt.ion()
        self.graphs = []
        for ylabel, title in labels:
            self.graphs.append(Graph(title, ylabel))

    def update(self, data: list[tuple]):
        for pos, (data_arr, gen) in enumerate(data):
            try:
                self.graphs[pos].update(data_arr, gen)
            except: ...


class MessageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global avg_performance_list
        global best_performance_list
        global genome_count_list
        global gen_performance_list
        global manager

        self.send_response(200)
        length = int(self.headers.get('Content-length', 0))
        data = json.loads(self.rfile.read(length).decode())
        avg_performance_list += [data["af"]]
        best_performance_list += [data["bf"]]
        genome_count_list += [data["gc"]]
        gen_performance_list += [data["gbf"]]
        generation = data["gen"]
        manager.update(((genome_count_list, generation),
                        (gen_performance_list, generation),
                        (best_performance_list, generation),
                        (avg_performance_list, generation)))

        print(f"Added data: {data}")


def neat_stats():
    global avg_performance_list
    global best_performance_list
    global genome_count_list
    global gen_performance_list
    global manager

    best_performance_list = []
    genome_count_list = []
    gen_performance_list = []
    avg_performance_list = []
    manager = Manager((("Genomes", "Genome Count"),
                       ("Fitness", "Best Fitness Per Generation"),
                       ("Fitness", "Best Fitness All Time"),
                       ("Fitness", "Average Fitness Per Generation")))

    server_address = ("127.0.0.1", 1234)
    httpd = HTTPServer(server_address, MessageHandler)
    print("Booted stats listener: waiting for requests")
    httpd.serve_forever()