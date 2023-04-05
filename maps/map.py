import folium
import osmnx as ox
import networkx as nx


# TODO: add markers and to show route between them
def draw_shortest_paths_for_points(G, map, source, target):
    # Find the nearest nodes to the source and target locations
    source_node = ox.distance.nearest_nodes(G, source[0], source[1])
    target_node = ox.distance.nearest_nodes(G, target[0], target[1])

    # Calculate the shortest path using NetworkX
    try:
        route = nx.shortest_path(G, source_node, target_node, weight='length')
    except nx.NetworkXNoPath:
        print("No path found between source and target")

    # Convert the route to a list of coordinates
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    # # Add a PolyLine to the map to show the route
    folium.PolyLine(locations=route_coords, color='blue').add_to(map)

    #   route_map = ox.plot_route_folium(G, route)

    # Add markers for the source and target locations
    folium.Marker(location=source).add_to(map)
    folium.Marker(location=target).add_to(map)

    # route_map.save('route.html')
    return map


locations = [[40.412061171927974, -3.7127956167260296], [40.40173481996409, -3.686531428185383],
             [40.43623693589483, -3.8343318617376467],[40.449301298454685, -3.7363132234585037]]

# Define the source and target locations
# source = (40.7128, -74.0060)
# target = (37.7749, -122.4194)
# Download the street network using OSMnx

def add_pin_to_place(map, coords):
    folium.Marker(location=coords).add_to(map)
    return map

def add_route_to_pins(map, source, target):
    # source_node = ox.distance.nearest_nodes(G, source[0], source[1])
    # target_node = ox.distance.nearest_nodes(G, target[0], target[1])
    # route = nx.shortest_path(G, source_node, target_node, weight='length')
    # route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    # # Add a PolyLine to the map to show the route
    folium.PolyLine(locations=[ source, target], color='red').add_to(map)

    return map


def add_pins_to_locations(map, locations):
    for i in range(len(locations)):
        source = locations[i]
        map = add_pin_to_place(map, source)

    return map

map = folium.Map(location=locations[0], zoom_start=12)
G = ox.graph_from_point(locations[0], dist=5000, network_type='drive')

map = add_pins_to_locations(map, locations)

for i in range(len(locations)-1):
    source = locations[i]
    target = locations[i + 1]

    map = add_route_to_pins(map, source, target)
    # target = locations[i+1]
    #
    # draw_shortest_paths_for_points(G, map, source, target)


# Save the map to an HTML file
map.save('map.html')