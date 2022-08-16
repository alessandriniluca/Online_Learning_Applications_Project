import numpy as np

class GraphWeightsEstimator():
    """This class is used to solve the step 5, where we must estimate the graph weights (they are the only uncertain parameters). We assumed that the 
    graph structure and the lambda is known.
    """
    def __init__(self, lamb, prod_number, products):
        """_summary_

        Args:
            lamb (float): lambda of the problem
            prod_number (int): number of products in the problem. In the assignment this number is fixed to 5
            products (list(product)): list of products in the problem
        """
        self.products = products
        self.lamb = lamb
        self.prod_number = prod_number
        # Initialization of the graph that will be updated after each round
        self.graph = np.zeros((prod_number, prod_number))
        self.seen = np.zeros((prod_number, prod_number))
        # Initialization of the number of users, that will be updated after each step
        self.user_number = 0
    
    def reset(self):
        """reset the estimation
        """
        self.graph = np.zeros((self.prod_number, self.prod_number))
        self.user_number = 0
        self.seen = np.zeros((self.prod_number, self.prod_number))
    
    def update_graph_probabilities(self, visits):
        """Update the graph estimated probability after each round

        Args:
            visits (list(list(int,int))): is a list of lists of tuples: for each user is saved the history of visits as tuples from a product to the subsequent
                as (prod, subsequent).
        """
        for user in visits:
            # increment the user
            self.user_number +=1
            for visit in user:
                start_product = visit[0]
                end_product = visit[1]
                clicked = visit[2]
                # Update the visits in the matrix
                if clicked:
                    self.graph[start_product][end_product]+=1
                self.seen[start_product][end_product]+=1
    
    def get_estimated_graph(self):
        """Return the estimation of graph weights

        Returns:
            np.array: returns the estimated matrix that before of this step was called "graph clicks"
        """
        # Compute the probability
        if(self.user_number == 0):
            return self.graph.copy()
        seen = self.seen.copy()
        seen[seen==0] = 1
        returned_graph = self.graph/seen
        return returned_graph