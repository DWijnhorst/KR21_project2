from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
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
        
    def Get_CPT(self, var):
        return self.bn.get_cpt(var)
    
    def Get_Vars(self):
        return self.bn.get_all_variables()    
    
############################################################################################# Methods to implement
    def Network_Pruning(self, query_vars, e):        
        #delete outgoing edges from nodes in e
        evidence_vars = e.index.values        
        for evidence in evidence_vars:            
            Outgoing_nodes = self.bn.get_children(evidence)
            for node in Outgoing_nodes:
                self.bn.del_edge((evidence, node))                         
                #apply factor reduction (zeroing out the instances that are incompatible with e)
                cpt = self.bn.get_cpt(node)
                new_cpt = self.bn.reduce_factor(e, cpt)
                self.bn.update_cpt(node, new_cpt)
                                               
        #delete any leaf nodes that do not appear in Q or e (iteratively)
        iteration = 0
        while iteration == 0 or LeafNodesPresent >= 1:
            iteration += 1
            LeafNodesPresent = 0
            vars = self.bn.get_all_variables()
            for query_var in query_vars:
                vars.remove(query_var)
            for evidence_var in evidence_vars:
                vars.remove(evidence_var)
            for var in vars:
                children = self.bn.get_children(var)
                if len(children) == 0:                        
                    LeafNodesPresent += 1 
                    self.bn.del_var(var)
    
    def D_separated(self, x,y,z):
        vars = self.bn.get_all_variables()
        leafnodes = []
        #delete every leaf node W not in x y or z
        for i in vars:
            if (len(self.bn.get_children(i)) == 0) and (i not in x) and (i not in y) and (i not in z):
                leafnodes.append(i)
        for leafnode in leafnodes:
            self.bn.del_var(leafnode)
        #recursive function call
        #delete every edge outgoing from nodes in z---> moet dit maar 1x????
        for var in z:
            if var in vars:
                childnodes = self.bn.get_children(var)
                if len(self.bn.get_children(var)) != 0:
                    for child in childnodes:
                        self.bn.del_edge((var , child))
        while not leafnodes:
            self.D_separated(x,y,z)
        graph = self.bn.get_interaction_graph()
        #see if exists a path
        for subx in x:
            for suby in y:
                if nx.has_path(graph, subx, suby) == False:
                    return True
        return False
    
    def Independence(self, x, y,z):
        if self.D_sperated(x,y,z) == True:
            return True
        else:
            return False
        #return independent == True / False
    
    def Marginalization(self, x):#f?
        cpt = self.bn.get_cpt(x)        
        #variables
        old_length = int(len(cpt.index))
        row_names = cpt.index.values
        uneven_indexes = []
        for index in range(0, old_length):
            if int(index) %2 == 1:
                uneven_indexes.append(index)        
        #update factors
        for index in range(0, old_length, 2):
            cpt['p'][index] = cpt['p'][index] + cpt['p'][index+1]         
        #drop every other row
        for index in uneven_indexes:
            cpt = cpt.drop(row_names[index], axis=0)            
        #drop column of var
        cpt = cpt = cpt.drop(x, axis=1)         
        return cpt
        #return CPT where x is summed-out 
    
    def Maxing_Out(self, x):
        cpt = self.bn.get_cpt(x)
        old_length = int(len(cpt.index))
        row_names = cpt.index.values
        uneven_indexes = []
        for index in range(0, old_length):
            if int(index) %2 == 1:
                uneven_indexes.append(index)
        for index in range(0, old_length, 2):
            if cpt['p'][index] > cpt['p'][index+1]:
                cpt['p'][index] = cpt['p'][index]
            else:
                cpt['p'][index] = cpt['p'][index+1]
        for index in uneven_indexes:
            cpt = cpt.drop(row_names[index], axis=0)        
        return cpt           
        # return (CPT where x is maxed-out) , (the instantiation of x which led to the maximized value)
    
    def Factor_Multiplication(self, f, g):
        f = f.tolist()
        g = g.tolist()
        h = []
        for factor1 in f:
            for factor2 in g:
                y = round(factor1*factor2,2)
                h.append(y)
        h = pd.Series(h)
        return h
        # return h (which is fg) 
           
    def Ordering(self,set_var, heuristic):
        #heuristic is min-degree or min-fill
        if heuristic == 'min-degree':
            ordering = []
            length = len(set_var) 
            graph = self.bn.get_interaction_graph()
            edges = graph.edges()                           
            while len(ordering) < length:            
                #fill ordering according to the smallest degree
                chosen_var = set_var[0]#set first var in set_var as default
                current_degree = 0                
                for edge in edges:
                        if chosen_var in edge:
                            current_degree += 1                                         
                for var in set_var: 
                    degree = 0 
                    for edge in edges:
                        if var in edge:
                            degree += 1
                    if degree < current_degree:#replace default var if current var has a smaller degree
                        chosen_var = var
                        current_degree = degree            
                ordering.append(chosen_var)#append the chosen var to the ordering                
                #Sum-out x from the interaction graph            
                for edge in edges:
                    if chosen_var in edge:
                        var1 = edge[0]
                        var2 = edge[1]
                        graph.remove_edge(var1, var2)
                graph.remove_node(chosen_var)
                set_var.remove(chosen_var) 
            #add last var               
            if set_var:
                ordering.append(set_var[0])                 
            return ordering  
        elif heuristic == 'min-fill':
            #implement second heuristic   
            pass           
        # return good ordering for elimination of set_var based on min-degree heuristics and min-fill heuristics
    
    def Variable_Elimination(self, input_net, set_var):
        pass
    # return input_net without set_var
    
    def Marginal_Distributions(self, q, e):
        self.Network_Pruning(self, q, e)
        cpt = self.bn.get_cpt(q)
        return cpt['p']
        # return marginal distribution P(q|e)
        
    def MAP(self, q, e):
        pass
    # return maximum a-posteriory instantion + value q
    
    def MEP(self, e):
        MPE = []
        #first apply network pruning given evidence e
        vars = self.bn.get_all_variables()
        self.Network_Pruning(vars, e)
        
        #get elimination order from ordering?
        ordering = self.Ordering(vars, "min-degree")
        
        #for each var: get max prob. and assign that truth value and update with this prob. factor 
        for var in ordering:  
            #get max cpt["p"]          
            cpt = self.bn.get_cpt(var)
            print(cpt)
            
            #set this truth value and append to MPE
            #for all cpt's containing var: apply factor multiplication of max * current p
                                
        pass
        # return most probable explanation given e

class main():
    # Variables
    Truth_value = True
    Var = "dog-out"
    Evidence = pd.Series({Var : Truth_value})
    Query_var = ["family-out"]
    x = ['bowel-problem', 'dog-out']
    y = ['hear-bark']
    z = ['family-out']
    
    #Init net
    NET = BNReasoner("testing/dog_problem.BIFXML") #initializing network)
    # NET = BNReasoner("testing/lecture_example.BIFXML") #initializing network)   
        
    #show NET 
    # NET.Draw()      
    
    #Applying network pruning
    NET.Network_Pruning(Query_var, Evidence)
    
    #Applying marginalization    
    Var2 = 'hear-bark'
    cpt = NET.Get_CPT(Var2)
    f = cpt['p']   
    # NET.Marginalization(f, Var2)
    
    #Applying MPE
    # NET.MEP(Evidence)      
    
    #finding a good ordering for variable elimination
    # NET.Ordering(NET.Get_Vars())
    
if __name__ == "__main__":
    main()