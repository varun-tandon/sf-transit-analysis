import pandas as pd
import networkx as nx
import numpy as np
import time
import pickle
from tqdm import tqdm 

def convert_time_to_seconds(time):
    time_split = time.split(':')
    return int(time_split[0]) * 3600 + int(time_split[1]) * 60 + int(time_split[2])

# Constants
LINES_TO_EXCLUDE = ['PH', 'PM']
SERVICE_ID = 78965
START_NODE = -1
END_NODE = -2
START_NODE_LAT_LONG = (37.772653, -122.426767)
END_NODE_LAT_LONG = (37.800468, -122.409065)
CURRENT_TIME = convert_time_to_seconds('12:00:00')

# Load in GTFS Data
trips_df = pd.read_csv('trips.txt')
stop_times_df = pd.read_csv('stop_times.txt')
stops_df = pd.read_csv('stops.txt')

# filter the trips by service_id
trips_df = trips_df[trips_df['service_id'] == SERVICE_ID]
# filter by route_id to remove 'PH' and 'PM' lines
trips_df = trips_df[~trips_df['route_id'].isin(LINES_TO_EXCLUDE)]
# get all the trip_ids for the selected lines
trip_ids = trips_df['trip_id'].tolist()
# filter stop_times by trip_id
stop_times_df = stop_times_df[stop_times_df['trip_id'].isin(trip_ids)]
# se the type of stop_id to int
stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(int)
# get all the stop_ids for the selected lines
stop_ids = stop_times_df['stop_id'].unique().tolist()

stops_in_graph_df = stops_df[stops_df['stop_id'].isin(stop_ids)]
# put the lat and long in a numpy array
stops_in_graph = stops_in_graph_df[['stop_id', 'stop_lat', 'stop_lon']].to_numpy()
# compute pairwise distances between all stops using numpy
distances = np.sqrt(np.sum((stops_in_graph[:, 1:] - stops_in_graph[:, 1:][:, np.newaxis]) ** 2, axis=-1))
# for each stop in the graph, find any stops within 150 ft and add an edge of 30 seconds
edges_dict = dict()
for i in range(len(stops_in_graph)):
    for j in range(len(stops_in_graph)):
        if i != j and distances[i][j] * 364000 < 350:
            edge_tuple = (stops_in_graph[i][0], stops_in_graph[j][0])
            edges_dict[edge_tuple] = [(-30, -30, 'walk')]

shared_dict = dict()
try:
    with open('shared_dict.pkl', 'rb') as f:
        shared_dict = pickle.load(f)
except:
    for trip_id in tqdm(trip_ids):
        trip_data = trips_df[trips_df['trip_id'] == trip_id]
        route_name = trip_data['route_id'].iloc[0]
        trip_stop_times = stop_times_df[stop_times_df['trip_id'] == trip_id]
        # sort by stop_sequence
        trip_stop_times = trip_stop_times.sort_values(by=['stop_sequence'])
        # for every adjacent pair of stops, create a tuple of the two stops
        # and add it to the edges_dict as a key with the value being a tuple (departure_time from first stop, arrival_time at second stop)
        for i in range(len(trip_stop_times) - 1):
            first_stop = trip_stop_times.iloc[i]
            second_stop = trip_stop_times.iloc[i + 1]
            # make both ints
            edge_tuple = (first_stop['stop_id'], second_stop['stop_id'])
            if edge_tuple not in shared_dict:
                shared_dict[edge_tuple] = []
            first_stop_departure_time = convert_time_to_seconds(first_stop['departure_time'])
            second_stop_arrival_time = convert_time_to_seconds(second_stop['arrival_time'])
            shared_dict[edge_tuple].append((first_stop_departure_time, second_stop_arrival_time, route_name))
    with open('shared_dict.pkl', 'wb') as f:
        pickle.dump(shared_dict, f)

# merge the shared_dict into the edges_dict
for key in shared_dict:
    if key not in edges_dict:
        edges_dict[key] = []
    edges_dict[key] += shared_dict[key]

# deduplicate the values in the edges_dict
for key in edges_dict:
    edges_dict[key] = list(set(edges_dict[key]))


walk_speed_ft_per_sec = 4.7
# find all the nodes that are within 20 mins of the start and end nodes
for node in stops_in_graph:
    node_id, node_lat, node_long = node
    distance_to_start_node = np.sqrt((node_lat - START_NODE_LAT_LONG[0]) ** 2 + (node_long - START_NODE_LAT_LONG[1]) ** 2) * 364000
    distance_to_end_node = np.sqrt((node_lat - END_NODE_LAT_LONG[0]) ** 2 + (node_long - END_NODE_LAT_LONG[1]) ** 2) * 364000
    if distance_to_start_node / walk_speed_ft_per_sec < 1200:
        time_to_walk = int(distance_to_start_node / walk_speed_ft_per_sec)
        edges_dict[(START_NODE, node_id)] = [(-time_to_walk, -time_to_walk, 'walk')]
    if distance_to_end_node / walk_speed_ft_per_sec < 1200:
        time_to_walk = int(distance_to_end_node / walk_speed_ft_per_sec)
        edges_dict[(node_id, END_NODE)] = [(-time_to_walk, -time_to_walk, 'walk')]

graph_nodes = stop_ids + [START_NODE, END_NODE]

unvisited = set()
node_distances = dict()
previous = dict()

for node in graph_nodes:
    unvisited.add(node)
    node_distances[node] = float('inf')
    previous[node] = None

node_distances[START_NODE] = 0
while len(unvisited) > 0:
    # find the unvisited node with the smallest distance
    min_distance = float('inf')
    min_node = None
    for node in unvisited:
        if node_distances[node] <= min_distance:
            min_distance = node_distances[node]
            min_node = node
    # if the min_node is the end_node, we are done
    if min_node == END_NODE:
        break
    # remove the node from the unvisited set
    unvisited.remove(min_node)
    # for each neighbor of the node, calculate the distance from the start node
    time_at_node = CURRENT_TIME + node_distances[min_node]
    # get all the neighbors of the min_node. neighbors are all the nodes that have an edge from the min_node
    neighbors = [edge[1] for edge in edges_dict if edge[0] == min_node]
    for neighbor in neighbors:
        edge_tuple = (min_node, neighbor)
        # find the next available edge that is after the time_at_node, with the shortest wait time
        min_wait_time = float('inf')
        min_wait_edge = None
        for edge in edges_dict[edge_tuple]:
            if edge[0] < 0 and min_wait_time > abs(edge[0]):
                min_wait_time = 0
                min_wait_edge = edge
            elif edge[0] >= time_at_node and edge[0] - time_at_node < min_wait_time:
                min_wait_time = edge[0] - time_at_node
                min_wait_edge = edge
        # if there is no available edge, continue
        if min_wait_edge is None:
            continue
        if min_wait_edge[0] < 0:
            edge_travel_time = abs(min_wait_edge[0])
        else:
            edge_travel_time = min_wait_edge[1] - min_wait_edge[0]
        alt = node_distances[min_node] + min_wait_time + edge_travel_time
        if alt < node_distances[neighbor]:
            node_distances[neighbor] = alt
            previous[neighbor] = (min_node, min_wait_edge[2])

# reconstruct the path, and the lines taken
path = []
current_node = END_NODE
while current_node is not None:
    path.append(current_node)
    current_node = previous[current_node][0] if previous[current_node] is not None else None
path.reverse()
lines = []
for i in range(len(path) - 1):
    lines.append(previous[path[i + 1]][1])
print(path)
print(lines)

# also compute the time taken to travel the path
time_taken = node_distances[END_NODE]
print(time_taken)

station_names = [ (stop_id, stops_df[stops_df['stop_id'] == stop_id]['stop_name'].iloc[0]) for stop_id in path if stop_id in stops_df['stop_id'].values ]
print(station_names)
