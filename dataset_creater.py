import pandas as pd
from scipy.special import erf
from scipy.linalg import inv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class cs_dataset:
    def __init__(self, file_path, cs_criteria):
        self.file_path = file_path
        self.cs_criteria =cs_criteria
        self.df = pd.read_csv(self.file_path, delimiter=',')
        self.sequence = self.df.groupby('NUM')['RES'].first().to_numpy()
        self.NCACO, self.y= self._gen_NCACO_array(self._generate_backbonecs())
        self.NCOCA = self._gen_NCOCA_array(self._generate_backbonecs())
        self.CANCO = self._gen_CANCO_array(self._generate_backbonecs())
        self.graph = []
        n=len(self.NCACO)
        self.rmsd_mtx=[[[] for _ in range(n)] for _ in range(n)]
        for i in range(len(self.NCACO)):
            graph, rmsd_mtx=self.build_tree_for_headnode(self.NCACO, self.NCOCA, self.CANCO,  self.rmsd_mtx, index= i, cut_off=self.cs_criteria)
            self.graph.append(graph)
            self.rmd_mtx=rmsd_mtx


    def _generate_backbonecs(self):
        # Filter rows for N, CA, C atoms
        filtered_df = self.df[self.df['ATOMNAME'].isin(['N', 'CA', 'C'])].copy()

        # Add random noise +/- 0.3 to SHIFT column
        noise = np.random.uniform(low=-0.3, high=0.3, size=len(filtered_df))
        filtered_df['SHIFT'] += noise  # Modify SHIFT column by adding noise

        # Create an error column with the absolute value of the noise
        filtered_df['error'] = np.abs(noise)

        # Add standard deviation column with random values between 0.2 and 1
        filtered_df['std'] = np.abs(noise) + np.random.uniform(low=0.1, high=0.3, size=len(filtered_df))

        return filtered_df

    # def _gen_NCACO_array(self, df):
    #     # Pivot the DataFrame to get each residue with N, CA, C and their std
    #     pivot_df = df.pivot(index=['RES', 'NUM'], columns='ATOMNAME').reset_index()
    #     pivot_df = pivot_df.sort_values(by='NUM').fillna(0)  # Fill NaN values with 0

    #     num_residues = pivot_df.shape[0]
    #     result_array = np.zeros((num_residues, 8))

    #     for i in range(len(pivot_df)):
    #         row = pivot_df.iloc[i]
    #         result_array[i, 0] = row[('SHIFT', 'N')]
    #         result_array[i, 1] = row[('std', 'N')]
    #         result_array[i, 2] = row[('SHIFT', 'CA')]
    #         result_array[i, 3] = row[('std', 'CA')]
    #         result_array[i, 4] = row[('SHIFT', 'C')]
    #         result_array[i, 5] = row[('std', 'C')]
    #         result_array[i, 6] = 0
    #         result_array[i, 7] = i 
    #     return np.round(result_array, 1)

    def _gen_NCACO_array(self, df):
        # Pivot the DataFrame to get each residue with N, CA, C and their std
        pivot_df = df.pivot(index=['RES', 'NUM'], columns='ATOMNAME').reset_index()
        
        # Randomize the sequence of residues
        pivot_df = pivot_df.sample(frac=1).reset_index(drop=True).fillna(0)

        num_residues = pivot_df.shape[0]
        result_array = np.zeros((num_residues, 8))
        residue_info = []

        for i in range(len(pivot_df)):
            row = pivot_df.iloc[i]
            
            # Populate the result array with SHIFT and std values for N, CA, and C
            result_array[i, 0] = row[('SHIFT', 'N')]
            result_array[i, 1] = row[('std', 'N')]
            result_array[i, 2] = row[('SHIFT', 'CA')]
            result_array[i, 3] = row[('std', 'CA')]
            result_array[i, 4] = row[('SHIFT', 'C')]
            result_array[i, 5] = row[('std', 'C')]
            result_array[i, 6] = 0
            result_array[i, 7] = i 

            # Append only the residue number and residue type as a tuple
            residue_info.append((int(row['NUM']), str(row['RES'].values[0])))

        return np.round(result_array, 1), residue_info


    def _gen_NCOCA_array(self, df):
        pivot_df = df.pivot(index=['RES', 'NUM'], columns='ATOMNAME').reset_index()
        pivot_df = pivot_df.sort_values(by='NUM').fillna(0)

        num_residues = pivot_df.shape[0]
        # Update the result array to have 8 columns (7 original + 1 for index)
        result_array = np.zeros((num_residues, 8))

        for i in range(len(pivot_df) - 1):
            row = pivot_df.iloc[i]
            next_row = pivot_df.iloc[i + 1]

            # Assign values to the result array
            result_array[i, 0] = next_row[('SHIFT', 'N')]
            result_array[i, 1] = next_row[('std', 'N')]
            result_array[i, 2] = row[('SHIFT', 'CA')]
            result_array[i, 3] = row[('std', 'CA')]
            result_array[i, 4] = row[('SHIFT', 'C')]
            result_array[i, 5] = row[('std', 'C')]
            result_array[i, 6] = 1
            
            # Add the original index of the current row to the last column
            result_array[i, 7] = i  # Get the index of the current row from pivot_df

        return np.round(result_array, 1)


    def _gen_CANCO_array(self, df):
        pivot_df = df.pivot(index=['RES', 'NUM'], columns='ATOMNAME').reset_index()
        pivot_df = pivot_df.sort_values(by='NUM').fillna(0)

        num_residues = pivot_df.shape[0]
        result_array = np.zeros((num_residues, 8))

        for i in range(1, len(pivot_df)):
            row = pivot_df.iloc[i]
            pre_row = pivot_df.iloc[i - 1]

            result_array[i, 0] = row[('SHIFT', 'N')]
            result_array[i, 1] = row[('std', 'N')]
            result_array[i, 2] = row[('SHIFT', 'CA')]
            result_array[i, 3] = row[('std', 'CA')]
            result_array[i, 4] = pre_row[('SHIFT', 'C')]
            result_array[i, 5] = pre_row[('std', 'C')]
            result_array[i, 6] = 2
            result_array[i, 7] = i 

        return np.round(result_array, 1)

    @staticmethod
    def calculate_rmsd(vec1, vec2, indices):
        diffs = [(vec1[i] - vec2[i]) ** 2 for i in indices]
        rmsd = np.sqrt(np.mean(diffs))
        return rmsd

    @staticmethod
    def calculate_3d_overlap(mu1, cov1, mu2, cov2):
        mu1, mu2 = np.array(mu1), np.array(mu2)
        cov1, cov2 = np.array(cov1), np.array(cov2)
        combined_cov = cov1 + cov2
        diff = mu1 - mu2
        d = np.sqrt(diff.T @ inv(combined_cov) @ diff)
        return erf(d / np.sqrt(2))
    

    def build_tree_for_headnode(self, a, b, c, rmsd_mtx, index, cut_off):
        # Create a directed graph
        G = nx.DiGraph()
        # Define the head node from array 'a'
        head_idx=index
        head_node = a[index]
        head_id = tuple(head_node)  # Use tuple as a unique identifier for the node
        G.add_node(head_id, label='a', embedding=head_node)
        
        # Right Tree - follow order a -> c -> b -> a
        for c_node in c:
            rmsd_ac = self.calculate_rmsd(head_node, c_node, [0, 2])
            if rmsd_ac < cut_off:
                c_id = tuple(c_node)
                G.add_node(c_id, label='c', embedding=c_node)
                G.add_edge(head_id, c_id, rmsd=rmsd_ac, tree='right')
                
                for b_node in b:
                    rmsd_cb = self.calculate_rmsd(c_node, b_node, [0, 4])
                    if rmsd_cb < cut_off:
                        b_id = tuple(b_node)
                        G.add_node(b_id, label='b', embedding=b_node)
                        G.add_edge(c_id, b_id, rmsd=rmsd_cb, tree='right')
                        
                        for a_subnode in a:
                            rmsd_ba = self.calculate_rmsd(b_node, a_subnode, [2, 4])
                            if rmsd_ba < cut_off:
                                a_sub_id = tuple(a_subnode)
                                G.add_node(a_sub_id, label='a', embedding=a_subnode)
                                G.add_edge(b_id, a_sub_id, rmsd=rmsd_ba, tree='right')
                                a_sub_idx = int(a_subnode[7])
                                rmsd_mtx[head_idx][a_sub_idx].append(rmsd_ac + rmsd_cb + rmsd_ba)
                                

        # Left Tree - follow order a -> b -> c -> a
        for b_node in b:
            rmsd_ab = self.calculate_rmsd(head_node, b_node, [2, 4])
            if rmsd_ab < cut_off:
                b_id = tuple(b_node)
                G.add_node(b_id, label='b', embedding=b_node)
                G.add_edge(head_id, b_id, rmsd=rmsd_ab, tree='left')
                
                for c_node in c:
                    rmsd_bc = self.calculate_rmsd(b_node, c_node, [0, 4])
                    if rmsd_bc < cut_off:
                        c_id = tuple(c_node)
                        G.add_node(c_id, label='c', embedding=c_node)
                        G.add_edge(b_id, c_id, rmsd=rmsd_bc, tree='left')
                        
                        for a_subnode in a:
                            rmsd_ca = self.calculate_rmsd(c_node, a_subnode, [0, 2])
                            if rmsd_ca < cut_off:
                                a_sub_id = tuple(a_subnode)
                                G.add_node(a_sub_id, label='a', embedding=a_subnode)
                                G.add_edge(c_id, a_sub_id, rmsd=rmsd_ca, tree='left')
                                a_sub_idx = int(a_subnode[7])
                                rmsd_mtx[a_sub_idx][head_idx].append(rmsd_ab + rmsd_bc + rmsd_ca)
        
        return G, rmsd_mtx

    
    


    


    # def build_connect(self.NCACO, self.NCOCA, self.CANCO, cs_criteria):
    #     num_peaks=self.NCACO.shape[0]
    #     connection_matrix= torch(zeros) shape(num_peaks, num_peaks)
        

# def generate_limited_connection_matrix(graph, nodes_array, max_depth=3):
#     # Number of nodes
#     n = len(nodes_array)
#     connection_matrix = np.zeros((n, n))

#     # Convert node vectors to tuples to match identifiers in the graph
#     node_ids = [tuple(node) for node in nodes_array]

#     for i, source_node in enumerate(node_ids):
#         for j, target_node in enumerate(node_ids):
#             if i != j:
#                 # Initialize minimum path RMSD to None (indicating no path found yet)
#                 min_path_rmsd = None
                
#                 # Iterate through paths with length ≤ max_depth
#                 for path in nx.all_simple_paths(graph, source=source_node, target=target_node, cutoff=max_depth):
#                     # Calculate the total RMSD for this path
#                     path_rmsd = sum(graph[u][v]['rmsd'] for u, v in zip(path[:-1], path[1:]))
                    
#                     # Update minimum path RMSD if this path's RMSD is lower
#                     if min_path_rmsd is None or path_rmsd < min_path_rmsd:
#                         min_path_rmsd = path_rmsd

#                 # If a path within max_depth was found, update the matrix
#                 if min_path_rmsd is not None:
#                     connection_matrix[i, j] = min_path_rmsd

#     return connection_matrix

# # Example usage
# connection_matrix = generate_limited_connection_matrix(G, NCACO_array, max_depth=3)


# 

class TreeVisualizer:
    def __init__(self, head_nodes):
        self.head_nodes = head_nodes  # The list of head nodes (NCACO array)

    def _hierarchy_pos(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = self._hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                          xcenter=nextx, pos=pos, parent=root)
        return pos

    def hierarchy_pos(self, G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        return self._hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def plot_tree(self, G, index, cut_off=0.6):
        # Extract the specified head node from head_nodes list
        head_node = tuple(self.head_nodes[index])

        # Generate the tree layout
        pos = self.hierarchy_pos(G, root=head_node)

        # Define labels for nodes
        labels = {node: f"{data['label']} {data['embedding'][[0,2,4,6]]}" for node, data in G.nodes(data=True)}

        # Assign colors based on the node type
        colors = []
        for node, data in G.nodes(data=True):
            if data['label'] == 'a':
                colors.append('skyblue')
            elif data['label'] == 'b':
                colors.append('lightgreen')
            elif data['label'] == 'c':
                colors.append('lightcoral')

        plt.figure(figsize=(15, 10))

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700)
        # Draw the labels
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black")

        # Separate and color edges based on tree type
        right_edges = [(u, v) for u, v, d in G.edges(data=True) if d['tree'] == 'right']
        left_edges = [(u, v) for u, v, d in G.edges(data=True) if d['tree'] == 'left']

        nx.draw_networkx_edges(G, pos, edgelist=right_edges, edge_color='blue')
        nx.draw_networkx_edges(G, pos, edgelist=left_edges, edge_color='red')

        plt.show()







    








    
