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
        
    def Draw(self):
        self.bn.draw_structure() 
    
    
############################################################################################# Methods to implement
    def Network_Pruning(self, query_var, e):
        #delete outgoing edges from nodes in e
        evidence_vars = e.index.values        
        for evidence in evidence_vars:            
            Outgoing_nodes = self.bn.get_children(evidence)
            for node in Outgoing_nodes:
                self.bn.del_edge((evidence, node)) 
                        
                #apply factor reduction (zeroing out the instances that are incompatible with e)
                cpt = self.bn.get_cpt(node)
                self.bn.reduce_factor(e, cpt)
                                
            #delete any leaf nodes that do not appear in Q or e (iteratively)
            iteration = 0
            while iteration == 0 or LeafNodesPresent >= 1:
                iteration += 1
                LeafNodesPresent = 0
                vars = self.bn.get_all_variables()
                vars.remove(query_var)
                vars.remove(evidence)
                for var in vars:
                    children = self.bn.get_children(var)
                    if len(children) == 0:                        
                        LeafNodesPresent += 1 
                        self.bn.del_var(var) 
            
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
    Truth_value = True
    Var = "dog-out"
    Evidence = pd.Series({Var : Truth_value})
    Query_var = "family-out"
    
    #Init net
    NET = BNReasoner("testing/dog_problem.BIFXML") #initializing network)
    
    #show NET --> works
    NET.Draw()      
    
    #NET.Network_Pruning(Query_var, Evidence)
    
if __name__ == "__main__":
    main()