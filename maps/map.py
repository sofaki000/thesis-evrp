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

    route_map.save('route.html')
    return map


locations = [[40.7128, -74.0060], [37.7749, -122.4194]]
# Define the source and target locations
# source = (40.7128, -74.0060)
# target = (37.7749, -122.4194)
# Download the street network using OSMnx

for i in range(len(locations)-1):
    source = locations[i]
    target = locations[i+1]
    G = ox.graph_from_point(source, dist=5000, network_type='drive')

    # Create a map centered at the source location
    map = folium.Map(location=source, zoom_start=12)
    draw_shortest_paths_for_points(G, map, source, target)


# Save the map to an HTML file
map.save('map.html')