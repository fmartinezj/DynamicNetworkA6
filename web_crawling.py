
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
import dwave_networkx as dnx 
import scrapy
from scrapy.crawler import CrawlerProcess
import json

#constructs a scrapy crawler object to be used for the web scraping process
class MySpider(scrapy.Spider):
    name = 'my_spider'

    #initializes it
    def __init__(self, domain, start_urls, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.allowed_domains = [domain]
        self.start_urls = start_urls

    #defines how it will parse the web pages
    def parse(self, response):
        links = response.css('a::attr(href)').getall()
        for link in links:
            if link.startswith(self.allowed_domains[0]):
                yield {
                    'source': response.url,
                    'target': link
                }

#function for crawling the web pages from the provided input file
def crawl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.readlines()
        n = int(data[0])
        domain = data[1].strip()
        start_urls = [url.strip() for url in data[2:]]

    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': output_file
    })
    process.crawl(MySpider, domain=domain, start_urls=start_urls)
    process.start()


#tbd computes the pagerank of the web page nodes
def compute_pagerank(G):
    return nx.pagerank(G)


#parses crawled data from the output file to be used in order to generate graph
def parse_crawled_data(output_file):
    # Parse the crawled data from the JSON output file
    with open(output_file, 'r') as file:
        crawled_data = json.load(file)
    return crawled_data



#reads graph if there is one saved locally
def read_graph(file_name):
    G = nx.Graph()
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split()  # whitespace seperates the individual parts
            if parts:  # as long as line not empty
                source = parts[0]
                targets = parts[1:]
                for target in targets:
                    G.add_edge(source, target)
    return G

def read_digraph(file_name):
    G = nx.Digraph()
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split()  # whitespace seperates the individual parts
            if parts:  # as long as line not empty
                source = parts[0]
                targets = parts[1]
                a = parts[2]
                b = parts[3]
                for target in targets:
                    G.add_edge(source, target, a=a, b=b)
    return G
#saves randomaly generated graphs to local memory so that it can be used to plot and compute the shortest paths
def save_graph(G, file_name):
    with open(file_name, 'w') as file:
        for source, targets in G.adjacency():
            line = str(source) + ' ' + ' '.join(map(str,targets)) + '\n'
            file.write(line)

def save_digraph(G, file_name):
    with open(file_name, 'w') as file:
        for u, v, data in G.edges(data=True):
            a = data.get('a', 0)
            b = data.get('b', 0)

            line = f"{u} {v} {a} {b}\n"
            file.write(line)

#creates a random graph given the number of nodes and constant factor for generation
def create_random_graph(n, c):
    p = (c*np.log(n))/n  #p is the probability that the nodes are connected to each other
    G = nx.erdos_renyi_graph(n, p) #here is where we use the actual erdos reny graph function from nx
    return G

#creates random bipartite graph
def create_bipartite_graph(n, m, p):
    G = bipartite.random_graph(n,m,p)
    return G

#shows the shortest path from the chosen source node to the target nodes
#also displays it in the graph (select option 5 after computing the shortest path)
def shortest_path(G, source, target):
    #using shortest path function from nx
    shortestPath = nx.shortest_path(G, source=source, target=target)
    print(f"Shortest path from {source} to {target}: {shortestPath}")
    return shortestPath

# displays the graph that is loaded locally, whether the most recent randomly generated or the most recently saved graph
def plot_graph(G, path=None, plot_cluster_coeff=False, plot_neighborhood_overlap=False, overlap_threshold = 0.5, max_pixel=1500, min_pixel=500, directed_graph=False):
    pos = nx.spring_layout(G)  # I am getting the postion using spring_layout functionality
    # nx.draw(G, pos, with_labels=True) #draw graph relative to position and include labels
    
    if plot_cluster_coeff:
        clustering = nx.clustering(G)
        cluster_min = min(clustering.values())
        cluster_max = max(clustering.values())

        # sizes = {v: min_pixel + (max_pixel - min_pixel) * ((cv - cluster_min) / (cluster_max - cluster_min)) for v, cv in clustering.items()}
        # Adjusted computation of sizes to handle division by zero
        sizes = {}
        if cluster_max - cluster_min == 0:  # All clustering coefficients are the same
            uniform_size = min_pixel + (max_pixel - min_pixel) / 2  # Use an average size
            sizes = {v: uniform_size for v in G.nodes()}
        else:
            sizes = {v: min_pixel + (max_pixel - min_pixel) * ((cv - cluster_min) / (cluster_max - cluster_min)) for v, cv in clustering.items()}
        colors = [(cv - cluster_min) / (cluster_max - cluster_min) * 254 for cv in clustering.values()]
    else:
        sizes = [300 for _ in G.nodes()]  # Default size
        colors = 'skyblue'  # Default color

    nx.draw(G, pos, with_labels=True, node_size=[sizes[v] for v in G.nodes()], node_color=colors if plot_cluster_coeff else 'skyblue')

    if path: #activates when running shortest path computation so that it displays the red line (i.e. the shortest path)
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2, style='dotted')

    overlap_edges = []

    # Highlighting edges based on neighborhood overlap (if enabled)
    if plot_neighborhood_overlap:
        # Compute and plot neighborhood overlap for each edge
        for u, v in G.edges():
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            overlap = len(neighbors_u & neighbors_v) / len(neighbors_u | neighbors_v) if len(neighbors_u | neighbors_v) > 0 else 0
            # Here you could decide on a threshold for highlighting or use overlap value directly for edge color/intensity
            if overlap > overlap_threshold:  # Check if overlap exceeds the threshold
                overlap_edges.append((u, v))

        # Highlight edges with significant overlap
        nx.draw_networkx_edges(G, pos, edgelist=overlap_edges, edge_color='orange', width=2)

    if directed_graph:
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = 'blue', node_size = 500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=overlap_edges, edge_color='blue', arrows=True, arrowstyle='-|>')
        # nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)

    plt.show() #display the actual graph






# Function to plot the directed graph and add edge labels as well as calculate the nash equib and social optimal
def plot_directed_graph(n):
    # Creating a directed graph
    G = nx.DiGraph()
    
    n=n

    #old version
    # edges_with_coefficients = [
    #     (0, 1, {'a': 2, 'b': 0}),  # (source, target, {attributes})
    #     (0, 2, {'a': 0, 'b': 4}),
    #     (1, 2, {'a': 0, 'b': 0}),
    #     (1, 3, {'a': 0, 'b': 4}),
    #     (2, 3, {'a': 2, 'b': 0})
    # ]

    # G.add_edges_from(edges_with_coefficients)



    # Step 2: Add edges with capacities and costs
    G.add_edge(0, 1, capacity=n, weight=2*n, poly = '2x + 0')  # Edge from source 's' to node 'a' with capacity 3 and cost 1
    G.add_edge(0, 2, capacity=n, weight=4, poly = '0x + 4')  # Edge from source 's' to node 'b' with capacity 2 and cost 2
    G.add_edge(1, 2, capacity=n, weight=0, poly = '0x + 0')  # Edge from node 'a' to node 'c' with capacity 3 and cost 3
    G.add_edge(1, 3, capacity=n, weight=4, poly = '0x + 4')  # Edge from node 'b' to node 'c' with capacity 4 and cost 1
    G.add_edge(2, 3, capacity=n, weight=2*n, poly = '2x + 0')  # Edge from node 'b' to node 'd' with capacity 2 and cost 2
    
    # calculates the nash equilibrium
    mincostFlow = nx.max_flow_min_cost(G, 0, 3)
    mincost = nx.cost_of_flow(G, mincostFlow)

    print()

    
    # calculates the social optimal, taking into account different driver numbers
    socialOptimal1 = int(n/2) * ((2*n)+4) * int(n/2)
    socialOptimal2 = int(n%2) * ((2*n)+4) * int(n%2)
    socOpt = socialOptimal1 + socialOptimal2

    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='black')

    # Prepare and draw edge labels
    edge_labels = {(u, v): data['poly'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=10)

    # sets up the graph to be plotted
    plt.text(-0.75, 1, f'Nash Equilibrium: {mincost}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(-0.75, 0.8, f'Social Optimal: {socOpt}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(-0.75, 0, f'Number of Drivers: {n}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    print(f'Nash Equilibrium: {mincost}')
    print(f'Social Optimal: {socOpt}')
    print(f'Number of Drivers: {n}')
    plt.show()


def plot_bipartite_graph(G):
    pos = nx.bipartite_layout(G, nodes=[n for n, d in G.nodes(data=true) if d['bipartite']==0])
    nx.draw(G, pos, with_labels=True)
    plt.show()

def plot_preferred_seller_graph(assignments, valuations, prices):
    G = nx.DiGraph()
    for i, (buyer, valuation) in enumerate(zip(assignments, valuations)):
         G.add_edge(f"House {i+1}", f"Buyer {buyer+1}", weight=valuation[i])
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_network_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show() 

#from the instructions provided
def karate_club_graph():
    G = nx.karate_club_graph()
    print("Node Degree")
    for v in G:
        # print(f"{v:1000} {G.degree(v):1000}")
        print(f"{v:4} {G.degree(v):6}")

    newG = nx.draw_circular(G, with_labels=True)
    plt.show()
    return G

#partitions the graph based on the number of components the user wishes to break it up into
def partition_graph(G, num_components):
    initial_edges = G.number_of_edges()
    initial_comps = nx.number_connected_components(G)  # Corrected variable name typo
    print("Condition check (Should be True to enter loop):", nx.number_connected_components(G) > num_components)
    current_connected_comps = nx.number_connected_components(G)

    while nx.number_connected_components(G) < num_components:
        print("Inside the loop - modifying the graph.")
        betweenness = nx.edge_betweenness_centrality(G)
        if not betweenness:  # Check if betweenness dictionary is empty
            print("No more edges to remove; the graph is fully disconnected.")
            break  # Exit the loop because there are no more edges to remove
        max_betweenness_edge = max(betweenness, key=betweenness.get)
        G.remove_edge(*max_betweenness_edge)
    
    final_edges = G.number_of_edges()
    final_comps = nx.number_connected_components(G)
    print(f"Initial edges: {initial_edges}, Final edges: {final_edges}")
    print(f"Initial comps: {initial_comps}, Final comps: {final_comps}, Requested comps: {num_components}")
    return G


#assigns colors in order to assess homophily 
def homophily(G, p):
# Here colors are assigned randomnly to each node with influence of p
    for node in G.nodes():
        G.nodes[node]['color'] = 'red' if np.random.rand() < p else 'blue'
    
    # Calculate + print the assortativity coefficient
    assortativity = nx.attribute_assortativity_coefficient(G, 'color')
    print(f"Assortativity coefficient: {assortativity}")
    
    return assortativity

#part of the graph balancing
def balance(G, p):
    # Assign signs to each edge
    for u, v in G.edges(): 
        G[u][v]['sign'] = '+' if np.random.rand() < p else '-'
    
    

    #print the edge signs for testing
    print("Edge signs:")
    for u, v, data in G.edges(data=True):
        print(f"{u}-{v}: {data['sign']}")
    
   
    return True  # or False based on your balance checking logic




def market_clearing(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        header = lines[0].split()
        n = int(header[0])  # Number of houses
        prices = np.array(list(map(float, header[1].split(','))))  # Prices
        valuations = np.array([list(map(float, line.split(','))) for line in lines[1:]])

    # Normal assignment for all but the last house
    assignments = np.argmax(valuations[:, :-1], axis=0)

    # Assign the last house to the last buyer manually
    last_buyer = valuations.shape[0] - 1  # Index of the last buyer
    last_house = n - 1  # Index of the last house
    assignments = np.append(assignments, last_buyer)

    # Calculate payoffs for each buyer based on the difference between their valuation and the house price
    payoffs = np.array([valuations[assignments[i], i] - prices[i] for i in range(n)])

    print("Market Clearing Results: ")
    for i, (buyer, payoff) in enumerate(zip(assignments, payoffs)):
        print(f"House {i+1} assigned to Buyer {buyer+1} with price ${prices[i]:.2f} and Payoff: ${payoff:.2f}")

    print("Now plotting the Preferred-Seller Graph.")
    print()
    plot_market_clearance_graph(prices, assignments, payoffs, valuations.shape[0])
    return assignments, prices, payoffs

def plot_market_clearance_graph(prices, assignments, payoffs, num_buyers):
    G = nx.Graph()

    num_houses = len(prices)
    houses = [f"House {i+1}" for i in range(num_houses)]
    buyers = [f"Buyer {j+1}" for j in range(num_buyers)]

    G.add_nodes_from(houses, bipartite=0, color='lightblue')  # Set A: Houses
    G.add_nodes_from(buyers, bipartite=1, color='lightgreen')  # Set B: Buyers

    for i, (house, buyer) in enumerate(zip(houses, assignments)):
        G.add_edge(house, buyers[buyer], weight=prices[i], payoff=payoffs[i])

    pos = {}
    pos.update((node, (1, index * 2)) for index, node in enumerate(houses))  # Stagger houses for clarity
    pos.update((node, (2, index * 2 + 1)) for index, node in enumerate(buyers))  # Stagger buyers for clarity

    nx.draw(G, pos, with_labels=True, node_color=[data['color'] for node, data in G.nodes(data=True)])
    edge_labels = {(u, v): f"Price: ${data['weight']:.2f}, Payoff: ${data['payoff']:.2f}" for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.show()



# Example usage:
# market_clearing('path_to_your_file.txt')




def perfect_matching(G):
    matching = bipartite.maximum_matching(G)
    return matching

# Example call (assuming data is available)
# plot_market_clearance_graph(prices, assignments, payoffs)


# assignment 6 starts here
def cascade_effect(G, m, q):

    # here we select m random initiators
    initiators = np.random.choice(G.nodes(), m, replace=False)
    influenced = set(initiators)  # to start, only the initiators are influenced

    # this part simulates the actual cascade effect
    next_influenced = set()
    while True:
        for node in influenced:
            for neighbor in G.neighbors(node):
                if neighbor not in influenced:


                    # here count how many neighbors are affected

                    influenced_neighbors = sum((n in influenced) for n in G.neighbors(neighbor))
                    # if the number exceeds the threshold --> add to the next round's influenced set
                    if influenced_neighbors / G.degree(neighbor) > q:
                        next_influenced.add(neighbor)
        if not next_influenced:  # no nodes left to be influenced at this point so end it
            break

        influenced.update(next_influenced)
        next_influenced.clear()

    # visualizing the actual graph with the cascase effect in place
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, nodelist=list(influenced), node_color='r', label='Influenced')
    nx.draw_networkx_nodes(G, pos, nodelist=list(set(G.nodes()) - influenced), node_color='g', label='Not Influenced')


    nx.draw_networkx_edges(G, pos)


    nx.draw_networkx_labels(G, pos)

    plt.legend()
    plt.show()


def simulate_sir(G, p, lifespan, shelter, r):
    # intialize the suscpetible, infected, and recovered nodes
    initial_infected = np.random.choice(list(G.nodes()), int(p * G.number_of_nodes()), replace=False)
    susceptible = set(G.nodes()) - set(initial_infected)

    infected = set(initial_infected)

    recovered = set()

    # here we simulate the SIR model over the user input lifespan
    for day in range(lifespan):
        #sets for each type of node here
        new_infected = set()

        new_recovered = set()

        # infection spread mechanism
        for node in list(infected):  # here have static list to avoid runtime error during modification
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and np.random.random() > shelter:
                    new_infected.add(neighbor)

        # recovery and vaccination portion
        for node in list(infected):  # convert to list to avoid modification during iteration
            if np.random.random() < 0.1:  # recovery rate
                new_recovered.add(node)

        # vaccination --> here we are iterating over a copy of susceptible node set
        for node in list(susceptible):
            if np.random.random() < r:
                if node in susceptible:  # check in case node state changed during the process
                    susceptible.remove(node)
                    recovered.add(node)

        # here we update states
        infected.update(new_infected)

        infected.difference_update(new_recovered)
        recovered.update(new_recovered)

        susceptible.difference_update(new_infected)  # Remove newly infected from susceptible

        
    
    # final visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=list(susceptible), node_color='g', label='Susceptible')


    nx.draw_networkx_nodes(G, pos, nodelist=list(infected), node_color='r', label='Infected')
    nx.draw_networkx_nodes(G, pos, nodelist=list(recovered), node_color='b', label='Recovered')
    nx.draw_networkx_edges(G, pos)

    nx.draw_networkx_labels(G, pos)
    plt.legend()
    plt.show()

    return G



#runs the program with a menu and try-catch cases to prevent program stops
def main():
    G = None  # Initialize G to None to handle cases where G is not yet defined
    path = None  # Initialize path to None
    
    # Initial state of plotting options
    plot_cluster_coeff = False
    plot_neighborhood_overlap = False
    overlap_threshold = 0.5  # Default value, adjust as needed
    p = 0.5 #default p value to initiate datatype

    output_file = 'output.json'
    output_graph_file = 'output_graph.gpickle'
    initial_domain = 'https://dblp.org'

    while True: #standard menu as instructed
        print("\nWelcome to the Main Menu:")
        print("1. Read Graph")
        print("2. Save graph")
        print("3. Create A Graph")
        print("4. Algorithms")
        print("5. Plot Graph (G)")
        print("6. Assign and Validate Values")
        print("7. Exit Program")
        choice = input("Enter Your Choice (1-7): ")

        if choice == '1':
            isDirected = input("Enter 1 if graph is directed or 2 if graph is undirected.")
            if isDirected == '1':
                file_name = input("Enter the filename to read from: ")
                try:
                    G = read_digraph(file_name)
                    print("Graph has been read successfully.")
                except Exception as e:
                    print(f"Error reading graph: {e}") #test
            elif isDirected == '2':
                file_name = input("Enter the filename to read from: ")
                try:
                    G = read_graph(file_name)
                    print("Graph has been read successfully.")
                except Exception as e:
                    print(f"Error reading graph: {e}") #test
            else:
                print("Invalid choice. Please enter 1 or 2 next time.")
            

        elif choice == '2':
            isDirected = input("Enter 1 if graph is directed or 2 if graph is undirected.")
            if isDirected == '1':
                print()
            elif isDirected == '2':
                if G is None: #must create or read graph before saving or computing shortest path
                    print("No graph has been found in memory. Please create or read a graph first.")
                    continue
                file_name = input("Enter the filename to save to: ")
                try:
                    save_graph(G, file_name)
                    print("Graph saved successfully.")
                except Exception as e:
                    print(f"Error saving graph: {e}")
            else: 
                print("Invalid choice. Please enter 1 or 2 next time.")


        elif choice == '3':
            print("\nWhat kind of graph would you like to create?")
            print("1) Random Graph (with parameters)")
            print("2) Karate Club Graph")
            print("3) Bipartite Graph")
            print("4) Market Clearance")
            print("5) Crawling (Create a Directed Graph)")
            subchoice = input("Enter Your Choice (1-5): ")
            if subchoice == '1':
                n = int(input("Enter the number of nodes (n): "))
                c = float(input("Enter the constant (c): "))
                G = create_random_graph(n, c)
                print("Random graph created successfully.")
            elif subchoice == '2':
                G = karate_club_graph()
                print("Karate Club graph created successfully.")
            elif subchoice == '3':
                n = int(input("Enter number of nodes in set A: "))
                m = int(input("Enter number of nodes in set B: "))
                p = int(input("Enter probability of an edge between the pairs of nodes. (0.0 - 1.0) -> "))
                G = create_bipartite_graph(n, m, p)
            elif subchoice == '4':
                filename = input("Enter the filename for your market clearance. (Hint: \"marketClearance.txt\") -> ")
                market_clearing(filename)
            elif subchoice == '5':
               
                crawl("crawlingFile", "output.json")
                crawled_data = parse_crawled_data(output_file)
                # G = construct_graph(crawled_data, initial_domain)

            else:
                print("Invalid graph choice. Please enter number 1-4.")

        elif choice == '4':
            if G is None:
                print("No graph in memory. Please create or read a graph first.")
                continue
            print("\nChoose an algorithm to compute: ")
            print("1) Shortest-Path")
            print("2) Partition G")
            print("3) Travel Equilibrium and Social Optimality")
            print("4) Market Clearance Algorithm")
            print("5) Compute PageRank")
            print("6) Plot PageRank Graph")
            print("7) Cascade Effect")
            print("8) COVID-19")
            subchoice = input("Enter Your Choice (1-7): ")
            if subchoice == '1':
                source = input("Enter the source node: ")
                target = input("Enter the target node: ")
                try:
                    path = shortest_path(G, source, target)
                except nx.NetworkXNoPath:
                    print(f"No path exists between {source} and {target}.")
                except Exception as e:
                    print(f"Error computing shortest path: {e}")
            elif subchoice == '2':
                Components = input("Enter the number (int) of components: ")
                numComponents = int(Components)
                
                G = partition_graph(G, numComponents)
                path = None
                # pos = nx.spring_layout(G)  # Layout for the graph
                # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='black', linewidths=1, font_size=10)
                # plt.show()
                # plot_graph(G, path)
            elif subchoice == '3':
                driverCount = input("Enter the number of drivers.")
            elif subchoice == '4':
                filename = input("Enter the filename for your market clearance. (Hint: \"marketClearance.txt\") -> ")
                market_clearing(filename)
            elif subchoice == '5':
                if G is None:
                    print("No graph in memory. Please create or read a graph first.")
                    continue
                pagerank = compute_pagerank(G)
                print("PageRank computed successfully.")
            elif subchoice == '6':
                if pagerank is None:
                    print("No PageRank computed yet. Please compute PageRank first.")
                    continue
                lower_bound = float(input("Enter the lower bound for PageRank scores: "))
                upper_bound = float(input("Enter the upper bound for PageRank scores: "))
                plot_pagerank_graph(G, pagerank, lower_bound, upper_bound)
            elif subchoice == '7':
                if G is None:
                    print("No karate graph in memory. Please create or read a karate graph first.")
                    continue
                m = int(input("Enter the number of initiators (int): "))
                q = float(input("Enter the threshold of the cascade (float): "))
                cascade_effect(G, m, q)
                print("PageRank computed successfully.")
            elif subchoice == '8':
                if G is None:
                    print("No graph in memory. Please create or read a graph first.")
                    continue
                p = float(input("Enter the fraction of initially infected nodes (float 0.0-1.0): "))
                lifespan = int(input("Enter the period of days of the simulation (int): "))
                shelter = float(input("Enter the fraction of the edges that are not considered by the shelter in-place (float 0.0-1.0): "))
                r = float(input("Enter the vaccination Rate (float 0.0-1.0): "))
                simulate_sir(G, p, lifespan, shelter, r)
                print("PageRank computed successfully.")
            else:
                print("Invalid choice. Please enter number 1-6.")

        elif choice == '5':
            
            # Submenu for plotting options
            print("\nGraph Plotting Options:")
            print("1) Toggle Cluster Coefficients")
            print("2) Toggle Neighborhood Overlap")
            print("3) Set Neighborhood Overlap Threshold")
            print("4) Shortest Path")
            print("5) Plot Undirected Graph")
            print("6) Plot Directed Graph")
            print("7) Preferred-Seller Graph")

            
            plot_choice = input("Enter your choice (1-7): ")
            # 
                # if G is None: #same as in step 2
                #     print("No graph has been found in memory. Please create or read a graph first.")
                #     continue
            if plot_choice == '1':
                plot_cluster_coeff = not plot_cluster_coeff
                print(f"Plotting Cluster Coefficients {'Enabled' if plot_cluster_coeff else 'Disabled'}")
            elif plot_choice == '2':
                plot_neighborhood_overlap = not plot_neighborhood_overlap
                print(f"Plotting Neighborhood Overlap {'Enabled' if plot_neighborhood_overlap else 'Disabled'}")
            elif plot_choice == '3':
                if plot_neighborhood_overlap:
                    overlap_threshold = float(input("Enter new overlap threshold (0-1): "))
                    print(f"Neighborhood Overlap Threshold set to {overlap_threshold}")
                else:
                    print("Neighborhood Overlap plotting is not enabled. Enable it first.")
            elif plot_choice == '4':
                source = input("Enter the source node: ")
                target = input("Enter the target node: ")
                try:
                    path = shortest_path(G, source, target)
                except nx.NetworkXNoPath:
                    print(f"No path exists between {source} and {target}.")
                except Exception as e:
                    print(f"Error computing shortest path: {e}")
            elif plot_choice == '5':
                plot_graph(G, path, plot_cluster_coeff, plot_neighborhood_overlap, overlap_threshold)
            elif plot_choice == '6':
                G = create_random_graph(1, 1)
                print("Ready to create directed graph.")
                n = input("Enter number of drivers:  ")
                try:
                    n = int(n)
                    plot_directed_graph(n)
                except ValueError:
                    print("Invalid input. Please enter a valid integer such as '4' or '10'.")
            elif plot_choice == '7':
                filename = input("Enter the filename for your market clearance. (Hint: \"marketClearance.txt\") -> ")
                market_clearing(filename)

        elif choice == '6':
           # Submenu for assign and validate options
            print("\nAssign and Validate Options:")
            print("1) Homphily")
            print("2) Balanced Graph")
            attribute_choice = input("Enter your choice (1 or 2): ") #user given option to enter their own p value
            if attribute_choice == '1':
                p = input("Enter p value between 0.0-1.0: ")
                try:
                    p = float(p)
                    if 0.0 <= p <= 1.0:
                        homophily(G, p)
                    else:
                        print("p value must be between 0.0 and 1.0.")
                except ValueError:
                    print("Invalid input. Please enter a valid floating-point number for p.")
            elif attribute_choice == '2':
                p = input("Enter p value between 0.0-1.0: ")
                try:
                    p = float(p)
                    if 0.0 <= p <= 1.0:
                        balance(G, p)
                    else:
                        print("p value must be between 0.0 and 1.0.")
                except ValueError:
                    print("Invalid input. Please enter a valid floating-point number for p.")
                    
            else:
                print("Invalid choice. Please enter number 1 or 2")



        elif choice == '7':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == "__main__":
    main()