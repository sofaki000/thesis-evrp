# https://medium.com/analytics-vidhya/measure-driving-distance-time-and-plot-routes-between-two-geographical-locations-using-python-39995dfea7e
API_KEY = "5b3ce3597851110001cf6248c652dcafb8d04ed7ab07b5295443b676"

import openrouteservice
from openrouteservice import convert
import folium
import json



# res = client.directions(coords)
# geometry = client.directions(coords)['routes'][0]['geometry']
# decoded = convert.decode_polyline(geometry)
#
# distance_txt = "<h4> <b>Distance :&nbsp" + "<strong>"+str(round(res['routes'][0]['summary']['distance']/1000,1))+" Km </strong>" +"</h4></b>"
# duration_txt = "<h4> <b>Duration :&nbsp" + "<strong>"+str(round(res['routes'][0]['summary']['duration']/60,1))+" Mins. </strong>" +"</h4></b>"
#
# folium.GeoJson(decoded).add_child(folium.Popup(distance_txt+duration_txt,max_width=300)).add_to(m)


def add_route_to_map(map, coord1, coord2):
    coords2 = [(coord1[0], coord1[1]), (coord2[0], coord2[1])]

    # Request the route from the OpenRouteService API
    routes = client.directions(coordinates=coords2, radiuses=[2000, 2000],profile='driving-car', format='geojson')

    # Extract the route geometry from the response
    route_geometry = routes['features'][0]['geometry']

    # Create a folium PolyLine object to represent the route
    folium.PolyLine(locations=route_geometry['coordinates'], color='blue', weight=2.5, opacity=1).add_to(map)
    from folium.plugins import AntPath
    #antpath = AntPath(route_geometry['coordinates'], color='blue', weight=2, opacity=1)

    antpath = AntPath(
        locations=route_geometry['coordinates'],
        color='blue',
        weight=2,
        opacity=1,
        dash_array=[10, 20],
        pulse_color='#FFFFFF'
    ).add_to(map)
    return map


def add_marker_to_locations(map, locations):
    # folium.Marker(location=list(coords[0][::-1]), popup="Galle fort", icon=folium.Icon(color="green"), ).add_to(m)
    # folium.Marker(location=list(coords[1][::-1]), popup="Jungle beach", icon=folium.Icon(color="red"), ).add_to(m)

    for index in range(len(locations)):
        location = locations[index]
        folium.Marker(location=location, popup=f'{index} customer', icon=folium.Icon(color="red"), ).add_to(map)

    return map



client = openrouteservice.Client(key=API_KEY)
locations = [[39.3955009561007, 22.174111922508565], [39.27583291151942, 22.540158506086964],
             [39.28459605461964, 21.955238707585405],[39.23200075431948, 22.226943388179674]]

route_order = [ 2,   0, 3 ,1]

# coords = ((80.21787585263182,6.025423265401452),(80.23990263756545,6.018498276842677))
# coords = ((40.412061171927974,-3.7127956167260296),(40.43623693589483, -3.8343318617376467))

m = folium.Map(location=locations[0],zoom_start=10, control_scale=True)

# for i in range(len(locations)-1):
#     add_route_to_map(m, locations[i], locations[i+1])

for i in range(len(route_order)-1):
    start_index = route_order[i]
    end_index = route_order[i+1]

    start = locations[start_index]
    end = locations[end_index]
    add_route_to_map(m, start, end)

m = add_marker_to_locations(m, locations)
m.save('map.html')

from PIL import Image
import io
img_data = m._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save('image.png')