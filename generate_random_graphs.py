import random
from xml.dom import minidom


def generate_graph(n, m):
    """Generates a random graph on n vertices and m edges. First an empty
    graph on n vertices is generated. Then m edges are added by choosing the
    start and and vertex u and v randomly."""
    g = {x: set() for x in range(n)}
    current_m = 0
    while True:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u < v and v not in g[u]:
            g[u].add(v)
            current_m += 1
            if current_m == m:
                break
    return g


def write_graph(g, filename):
    """Writes a given graph g to the file filename. For this the graphml
     format is respected."""
    graphml = minidom.Document()
    xml = graphml.createElement('graphml')
    graphml.appendChild(xml)
    for x in g:
        child = graphml.createElement('node')
        child.setAttribute('id', 'n{0}'.format(x))
        xml.appendChild(child)
    edgenum = 0
    for u in g:
        for v in g[u]:
            child = graphml.createElement('edge')
            child.setAttribute('id', 'e{0}'.format(edgenum))
            child.setAttribute('source', 'n{0}'.format(u))
            child.setAttribute('target', 'n{0}'.format(v))
            xml.appendChild(child)
            edgenum += 1

    xml_str = graphml.toprettyxml(indent="\t")
    with open(filename, 'w') as f:
        f.write(xml_str)


def main():
    """Generates a random graph in increments of 10 for all vertex sizes
    between 10 and 500 with edge ratios between 1 and 10"""
    for n in range(10, 501, 10):
        for i in range(1, 11):
            m = n * i
            if n*(n-1)/2 < m:
                continue
            g = generate_graph(n, m)
            write_graph(g, "data/random/graph.{0}.{1}.graphml".format(n, m))
            print("Done Graph:", n, m)


if __name__ == "__main__":
    main()
