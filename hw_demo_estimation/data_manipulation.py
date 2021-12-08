import pandas as pd


def compute_directed_edges(edges_w_features):
    """
    Converts the undirected edge information into a directed format, by duplicating each edge and flipping the node
    attributes to make it point in the opposite direction. This makes computation from the viewpoint of each node
    simpler.
    :param edges_w_features:
    :return:
    """
    opposite = edges_w_features.copy()
    # flipping the attributes of the endpoints
    opposite[["smaller_id", "greater_id", "AGE_x", "AGE_y", "gender_x", "gender_y"]] = \
        opposite[["greater_id", "smaller_id", "AGE_y", "AGE_x", "gender_y", "gender_x"]]
    directed = pd.concat([edges_w_features, opposite], ignore_index=True)
    return directed


def add_nbrs_by_gender(nodes, directed_edges):
    """
    Adds one column for each gender to the nodes table, which contain the number of neighbors of the given gender
    for each ndoe. Unknown-gender neighbors are not counted into either gender.
    :param nodes: Node feature data as DataFrame
    :param directed_edges: Edge data as DataFrame
    :return: the nodes DataFrame with the columns 0_nbrs and 1_nbrs added to it
    """
    w_nbrs = nodes.copy()
    w_nbrs = w_nbrs.set_index("user_id")
    nbrs = compute_nbrs_with_gender(directed_edges, 0.0)
    w_nbrs = w_nbrs.merge(nbrs, on="user_id")
    nbrs = compute_nbrs_with_gender(directed_edges, 1.0)
    w_nbrs = w_nbrs.merge(nbrs, on="user_id")
    return w_nbrs


def compute_nbrs_with_gender(directed_edges, gender):
    """
    Counts the number of neighbors with the given gender for each node.
    :param directed_edges: directed edge information as a DataFrame
    :param gender: which gender the counted neighbors should have
    :return: A table containing a single column with the number of filtered neighbors.
    """
    nbrs = directed_edges[directed_edges["gender_y"] == gender].groupby("smaller_id").count()["greater_id"].to_frame()
    nbrs = nbrs.rename_axis("user_id").rename(columns={"greater_id": ("%d_nbrs" % gender)})
    return nbrs