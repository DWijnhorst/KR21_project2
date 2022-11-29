from typing import Union
from BayesNet import BayesNet
import pandas as pd
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
    # returns all variables
############################################################################################## Basics for testing
    def Variables_in_net(self):
        variables = self.bn.get_all_variables()
        return variables

    def Get_All_CPTs(self):
        CPTs = self.bn.get_all_cpts()
        return CPTs

    def Get_CPT(self, string):
        CPT = self.bn.get_cpt(string)
        return CPT

    def Get_network(self):
        Network = self.bn.get_interaction_graph
        return Network

    def GetCompatible(self, tuple,  CPT):
        CompatibleInst = self.bn.get_compatible_instantiations_table(tuple, CPT)
        return CompatibleInst
############################################################################################# Methods to implement
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

class main():

    # Variables
    evidence = True
    Q = "light-on"
    query_var = pd.Series({Q : evidence})
    #Init net
    NET = BNReasoner("/Users/daanwijnhorst/Documents/GitHub/KR21_project2/testing/dog_problem.BIFXML") #initializing network)
    #show all CPTs
    CPT = NET.Get_CPT("light-on")
    print(CPT)
    # get compatible CPTs with a tuple given
    GETCOMPATIBLE = NET.GetCompatible(pd.Series({"light-on" : True, "dog-out" : False}), CPT)
if __name__ == "__main__":
    main()