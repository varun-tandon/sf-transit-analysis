import multiprocessing as mp
import networkx as nx
import numpy as np
import time
import pickle
from tqdm import tqdm 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from matplotlib.patches import Rectangle
import datetime

def convert_time_to_seconds(time):
    time_split = time.split(':')
    return int(time_split[0]) * 3600 + int(time_split[1]) * 60 + int(time_split[2])

# Constants
LINES_TO_EXCLUDE = ['PH', 'PM']
SERVICE_ID = 78965
START_NODE = -1
END_NODE = -2
# START_NODE_LAT_LONG = (37.772653, -122.426767) # 228 Haight
# START_NODE_LAT_LONG = (37.761632, -122.443222)
WALK_SPEED_FT_PER_SEC = 4.3
SCAN_WIDTH = 0.0038
TIME_LIMIT = 30

def load_and_filter_GTFS_data():
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
    return trips_df, stop_times_df, stops_df, trip_ids, stop_ids
    
def construct_basic_graph():
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
    return stops_in_graph, edges_dict

def load_shared_graph_with_GTFS_data(existing_edges):
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
    for key in shared_dict:
        if key not in existing_edges:
            existing_edges[key] = []
        existing_edges[key] += shared_dict[key]

    # deduplicate the values in the edges_dict
    for key in existing_edges:
        existing_edges[key] = list(set(existing_edges[key]))

    return existing_edges

def construct_walking_edges_to_node(target_node_id, target_node_lat, target_node_long, edges_dict):
    edges_added = set()
    for node in stops_in_graph:
        node_id, node_lat, node_long = node
        distance_to_start_node = np.sqrt((node_lat - target_node_lat) ** 2 + (node_long - target_node_long) ** 2) * 364000
        if distance_to_start_node / WALK_SPEED_FT_PER_SEC < 1200:
            time_to_walk = int(distance_to_start_node / WALK_SPEED_FT_PER_SEC)
            edges_dict[(target_node_id, node_id)] = [(-time_to_walk, -time_to_walk, 'walk')]
            edges_added.add((target_node_id, node_id))
    return edges_dict, edges_added

def construct_distances_dict(start_node, graph_nodes, edges_dict, current_time):
    unvisited = set()
    node_distances = dict()
    previous = dict()
    for node in graph_nodes:
        unvisited.add(node)
        node_distances[node] = float('inf')
        previous[node] = None

    node_distances[start_node] = 0
    while len(unvisited) > 0:
        min_node = min(unvisited, key=lambda node: node_distances[node])
        min_distance = node_distances[min_node]
        # remove the node from the unvisited set
        unvisited.remove(min_node)
        # for each neighbor of the node, calculate the distance from the start node
        time_at_node = current_time + node_distances[min_node]
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

    return node_distances, previous

def process_point(latitude, longitude, bm, stops_in_graph, node_distances):
    xpt, ypt = bm(longitude, latitude)
    if not bm.is_land(xpt, ypt):
        return
    END_NODE_LAT_LONG = (latitude, longitude)
    best_time = float('inf')
    node_latitudes = np.array([node[1] for node in stops_in_graph])
    node_longitudes = np.array([node[2] for node in stops_in_graph])
    distances_to_end_node = np.sqrt((node_latitudes - END_NODE_LAT_LONG[0]) ** 2 + (node_longitudes - END_NODE_LAT_LONG[1]) ** 2) * 364000
    walk_times = distances_to_end_node / WALK_SPEED_FT_PER_SEC
    total_travel_times = np.array([node_distances[node[0]] + walk_time for node, walk_time in zip(stops_in_graph, walk_times)])
    best_time = np.min(total_travel_times)
    return (latitude, longitude), best_time

def format_heatmap_dataframe(point_travel_times):
    df = pd.DataFrame(list(point_travel_times.items()), columns=['Location', 'TravelTime'])
    df[['Latitude', 'Longitude']] = pd.DataFrame(df['Location'].tolist(), index=df.index)
    df = df.drop('Location', axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df['TravelTime'] = df['TravelTime'] / 60
    df.loc[df['TravelTime'] > TIME_LIMIT, 'TravelTime'] = np.nan
    return df

def show_heatmap(heatmap_df):
    imagery = OSM()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': imagery.crs})
    ax.set_extent([-122.513681, -122.361190, 37.708216, 37.810136])
    ax.add_image(imagery, 15)
    ax.coastlines(resolution='10m')
    
    # overlay the heatmap on top with opacity 0.3
    x = heatmap_df['Longitude'].values
    y = heatmap_df['Latitude'].values
    z = heatmap_df['TravelTime'].values
    
    # filter the NaN values
    x = x[~np.isnan(z)]
    y = y[~np.isnan(z)]
    z = z[~np.isnan(z)]
    
    # draw a rectangle for each point and color it based on the travel time
    cmap = matplotlib.colormaps['RdYlGn_r']
    for i in range(len(x)):
        ax.add_patch(Rectangle((x[i], y[i]), SCAN_WIDTH, SCAN_WIDTH, color=cmap(z[i] / TIME_LIMIT), alpha=0.7, linewidth=0.00001, transform=ccrs.PlateCarree()))
    
    # draw a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=TIME_LIMIT))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.05)
    cbar.set_label('Travel Time (minutes)')
    plt.title('Travel Time Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

def convert_seconds_to_hhmmss(seconds):
    return f'{int(seconds // 3600)}:{int((seconds % 3600) // 60)}:{int(seconds % 60)}'

def compute_score(heatmap_df):
    total_score = 0
    # give 3 points for every coordinate within 10 minutes of the target time
    # give 2 points for every coordinate within 20 minutes of the target time
    # give 1 point for every coordinate within 30 minutes of the target time
    # use pandas functions to speed up the computation
    total_score += 3 * heatmap_df.loc[heatmap_df['TravelTime'] <= 10, 'TravelTime'].count()
    total_score += 2 * heatmap_df.loc[(heatmap_df['TravelTime'] > 10) & (heatmap_df['TravelTime'] <= 20), 'TravelTime'].count()
    total_score += 1 * heatmap_df.loc[(heatmap_df['TravelTime'] > 20) & (heatmap_df['TravelTime'] <= 30), 'TravelTime'].count()
    return total_score

def process_time(time_to_scan, graph_nodes, edges_dict, stops_in_graph, bm):
    node_distances, previous = construct_distances_dict(START_NODE, graph_nodes, edges_dict, time_to_scan)
    latitudes = np.arange(37.708216, 37.810136, SCAN_WIDTH)
    longitudes = np.arange(-122.513681, -122.361190, SCAN_WIDTH)
    point_travel_times = dict()
    results = [process_point(latitude, longitude, bm, stops_in_graph, node_distances) for latitude in latitudes for longitude in longitudes]
    for result in results:
        if result is not None:
            point_travel_times[result[0]] = result[1]
    
    heatmap_df = format_heatmap_dataframe(point_travel_times)
    score = compute_score(heatmap_df)
    return score

if __name__ == '__main__':
    # scan over the entire map
    bm = Basemap(projection='merc',llcrnrlat=37.708216,urcrnrlat=37.810136,\
                llcrnrlon=-122.513681,urcrnrlon=-122.361190,resolution='i')
    trips_df, stop_times_df, stops_df, trip_ids, stop_ids = load_and_filter_GTFS_data()
    stops_in_graph, edges_dict = construct_basic_graph()
    edges_dict = load_shared_graph_with_GTFS_data(edges_dict)
    f = open('results_coarse.txt', 'a')
    for start_node_lat in np.arange(37.708216, 37.810136, 0.01):
        for start_node_long in tqdm(np.arange(-122.513681, -122.361190, 0.01)):
            # check if the start_node is land
            xpt, ypt = bm(start_node_long, start_node_lat)
            if not bm.is_land(xpt, ypt):
                continue
            start = time.time()
            edges_dict, edges_added = construct_walking_edges_to_node(START_NODE, start_node_lat, start_node_long, edges_dict)
            graph_nodes = stop_ids + [START_NODE, END_NODE]
            # scan in 10 min intervals from 6:00:00 to 10:00:00
            times_to_scan = np.arange(6 * 60 * 60, 10 * 60 * 60, 30 * 60)
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap(process_time, [(time_to_scan, graph_nodes, edges_dict, stops_in_graph, bm) for time_to_scan in times_to_scan])
            for i, result in enumerate(results):
                f.write(f'{start_node_lat},{start_node_long},{convert_seconds_to_hhmmss(times_to_scan[i])},{result}\n')

        # save the heatmap to an image file
        # show_heatmap(heatmap_df)
        # plt.savefig(f'images/heatmap_{i}.png')
        # plt.close()
