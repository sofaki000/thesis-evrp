from scipy.spatial import distance_matrix


def create_distance_matrix(points):
    return distance_matrix(points, points)


def get_routes(solution, routing, manager):
  """Get vehicle routes from a solution and store them in an array."""
  # Get vehicle routes and store them in a two dimensional array whose
  # i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route)

  return routes

def print_solution(tour) -> object:  
    plan_output = ''
    for i in range(len(tour)):
          if i == len(tour)-1:
            plan_output += ' {}'.format(tour[i])
          else:
            plan_output += ' {} ->'.format(tour[i])
    print(plan_output)
    return plan_output

def get_tour_length_from_distance_matrix(tour, distance_matrix):
    length = 0
    prev_index = tour[0]
    for i in range(len(tour)):
        index = tour[i]
        length += distance_matrix[index][prev_index]
        prev_index = index

    content = f'Distance travelled:{length}'
    print("----------------------")
    print(content)
    return content, length