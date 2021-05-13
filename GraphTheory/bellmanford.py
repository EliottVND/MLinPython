def bellman_ford_shortest_path(graph, start, end):
    result = []
    distance = {}
    for i in graph.keys():
        distance[i] = (float("inf"),None)
    distance[start] = (0,None)
    for i in graph.keys():
        for j in graph[i]:
            if j[0] not in distance.keys():
                distance[j[0]] = (float("inf"),None)
    for i in range(len(graph[start])):
        distance[graph[start][i][0]] = (float(graph[start][i][1]), start)
    for iterations in range(len(graph)-1): # Les n-1 itérations
        done = False
        for k in graph.keys(): # Chaque keys
            if k != start:
                for l in range(len(graph[k])):
                    if distance[k][0] + float(graph[k][l][1]) < distance[graph[k][l][0]][0]:
                        distance[graph[k][l][0]] = (distance[k][0] + float(graph[k][l][1]),k)
                        done = True
            else:
                done = True
        if done == False :
            break
    if distance[end][0] == float("inf"):
        return []
    
    for k in graph.keys(): # Chaque keys
        for l in range(len(graph[k])):
            if distance[k][0] + float(graph[k][l][1]) < distance[graph[k][l][0]][0]:
                return "Negative Cycle"
     
    parent = distance[end][1]
    result.append(end)
    result.append(parent)
    if parent != None:
        while distance[parent][1] != None:
            parent = distance[parent][1]
            result.append(parent)
    result = result[:-1]
    result.reverse()
    return result