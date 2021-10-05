import igraph as ig

from syn.helpers.treelstm.dataloader import Tree


def get_trees_from_string(trees_str: list) -> list:
    return [Tree.from_string(tree) for tree in trees_str]


def plot_tree_graph(g, layout_name, label_name):
    print(g)
    # g.vs['label'] = g_tree.vs['attention_weight']
    # g.vs['label'] = g_tree.vs['pos']
    g.vs['label'] = g.vs[label_name]
    color_dict = {'root': 'red', 'leaf': 'Burlywood', 'branch': 'pink'}
    g.vs['color'] = [color_dict[node_type] for node_type in g.vs['type']]
    g.vs['size'] = 30
    # g.vs['label_dist'] = 5
    g.vs['label_size'] = 10

    layout = g.layout(layout=layout_name)
    ig.plot(g, layout=layout)


def build_tree_graph(g, parent_id, tree):
    if g.vs.indices is not None and len(g.vs.indices) > 0:
        cur_id = max(g.vs.indices) + 1
    else:
        cur_id = 0

    g.add_vertices(1)
    g.vs[cur_id]['label'] = tree.label
    g.vs[cur_id]['pos'] = tree.pos
    g.vs[cur_id]['alias'] = tree.alias
    if cur_id == 0:
        g.vs[cur_id]['type'] = 'root'
    else:
        g.vs[cur_id]['type'] = 'leaf' if tree.is_leaf() else 'branch'

    if parent_id != cur_id:
        g.add_edges([(parent_id, cur_id)])

    if tree.children is not None and len(tree.children) == 1:
        p_id = cur_id
        build_tree_graph(g, p_id, tree.children[0])

    if tree.children is not None and len(tree.children) == 2:
        p_id = cur_id
        build_tree_graph(g, p_id, tree.children[0])
        p_id = cur_id
        build_tree_graph(g, p_id, tree.children[1])


def get_attention_vectors(trees_str):
    trees = get_trees_from_string(trees_str)

    # sentences_tokens = []
    sentences_attention_weights = []
    for tree in trees:
        g_tree = ig.Graph()
        build_tree_graph(g_tree, 0, tree)

        # tokens = []
        attention_weights = []
        root_vertex = g_tree.vs.find(type='root')
        for leaf in tree.leaves():
            context_weight = 0.0
            leaf_vertex = g_tree.vs.find(alias=leaf.alias)
            shortest_path = g_tree.get_shortest_paths(root_vertex, leaf_vertex)[0]
            shortest_path_context = shortest_path[:len(shortest_path) - 2]
            preterminal = shortest_path[len(shortest_path) - 2:len(shortest_path) - 1][0]
            if len(shortest_path) > 2:
                for i in shortest_path_context:
                    context_weight += g_tree.vs[i]['label']

            # tokens.append(leaf.label)
            attention_weights.append(
                [g_tree.vs[preterminal]['label'], context_weight / (len(shortest_path) - 2)]
            )

        # sentences_tokens.append(tokens)
        sentences_attention_weights.append(attention_weights)

    return sentences_attention_weights
