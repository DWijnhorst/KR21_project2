from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def Network_Pruning(self, query_var, e):
        pass
    #return pruned_network
    def D_separated(self, x,y,z):
        pass
    #return D_separated == True / False
    def Independence(self, x, y,z):
        pass
    #return independent == True / False
    def Marginalization(self, f, x):
        pass   
    #return CPT where x is summed-out 
    def Maxing_Out(self, f, x, ):
        pass    
    # return (CPT where x is maxed-out) , (the instantiation of x which led to the maximized value)
    def Factor_Multiplication(self, f, g):
        pass
    # return h (which is fg)
    def Ordering(self,set_var):
        pass
    # return good ordering for elimination of set_var basred on min-degree heuristics and min-fill heuristics
    def Variable_Elimination(self, input_net, set_var):
        pass
    # return input_net without set_var
    def Marginal_Distributions(self, q, e):
        pass
    # return marginal distribution P(q|e)
    def MAP(self, q, e):
        pass
    # return maximum a-posteriory instantion + value q
    def MEP(self, e):
        pass
    # return most probable explanation given e



if __name__ == "__main()__":
    print('hello world')
    net = BNReasoner("testing/lecture_example.BIFXML")