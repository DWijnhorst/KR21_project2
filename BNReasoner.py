from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy as copy
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
                new_cpt = self.bn.reduce_factor(e, cpt)#reduce p to 0                
                new_cpt = new_cpt.drop(evidence, axis=1)#remove parent from child (no edge anymore)                
                new_cpt = new_cpt[new_cpt['p'] != 0]#remove row with p=0
                self.bn.update_cpt(node, new_cpt)    
                
                                          
        #delete any leaf nodes that do not appear in Q or e (iteratively)
        iteration = 0
        if len(query_vars) != len(self.bn.get_all_variables()):#MPE check
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
        if self.D_separated(x,y,z) == True:
            return True
        else:
            return False
        #return independent == True / False
    
    def Marginalization(self, x):#f nodig?
        cpt = self.bn.get_cpt(x)        
        #variables
        old_length = int(len(cpt.index))
        row_names = cpt.index.values
        uneven_indexes = []
        for index in range(0, old_length):
            if int(index) %2 == 1:
                uneven_indexes.append(index)        
        #update factors (sum)
        for index in range(0, old_length, 2):
            cpt['p'][index] = cpt['p'][index] + cpt['p'][index+1]         
        #drop every other row
        for index in uneven_indexes:
            cpt = cpt.drop(row_names[index], axis=0)            
        #drop column of marginalized var
        cpt = cpt.drop(x, axis=1)         
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
        # f = f.tolist()
        # g = g.tolist()
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
            ordering = []
            originalgraph = copy.deepcopy(self.bn.get_interaction_graph())
            originalset_var = set_var
            nx.draw(originalgraph, with_labels=True, node_size=3000)
            plt.show()
            # cycle:
            for i in range(len(originalset_var)):
                neighbors = dict()
                fill_value_dict = dict()
                originaledges = originalgraph.edges()
                for var in set_var:
                    neighbors[var] = list(nx.neighbors(originalgraph, var))
                #THIS INNER LOOP WORKS, SO ONLY 1 thing left to do: fix the while loop
                for i in range(len(set_var)):
                    graph = copy.deepcopy(originalgraph)
                    edges = graph.edges()
                    #i = 3
                    fill_value = 0
                    print('set_var[i] =', set_var[i])
                    #1. Remove edges from/to to set_var[i]
                    for neighbor in neighbors[set_var[i]]:
                        try:
                            graph.remove_edge(set_var[i], neighbor)
                        except:
                            graph.remove_edge(neighbor, set_var[i])
                    #2. Remove set_var[i] from graph

                    graph.remove_node(set_var[i])

                    #3. try if there are still direct connections between the old neighbors of i
                    for neighbor in range(len(neighbors[set_var[i]])-1):
                        for neighbor2 in range(neighbor+1, len(neighbors[set_var[i]])):
                            listforval = neighbors.get(set_var[i])
                            value1 = listforval[neighbor]
                            value2 = listforval[neighbor2]
                            print('value1 = ', value1)
                            print('value2 = ', value2)                    
                            if graph.has_edge(value1, value2) == False:
                                fill_value +=1  #if not a connection, it needs an additional edge, thus fillvalue+1
                    fill_value_dict.update({set_var[i] :  fill_value})
                    # add key of smallest value of dictionary to the elimination queue
                eliminationkey = min(fill_value_dict, key = fill_value_dict.get)
                print('fill_value_dict =' , fill_value_dict)
                ordering.append(eliminationkey)
                print('ordering = ', ordering)
                
                #deleting actual edges and nodes and adding edges that dont exist yet
                for neighbor in neighbors[eliminationkey]:
                    try:
                        originalgraph.remove_edge(eliminationkey, neighbor)
                    except:
                        originalgraph.remove_edge(neighbor, eliminationkey)
                
                originalgraph.remove_node(eliminationkey)

                for neighbor in range(len(neighbors[eliminationkey])-1):
                    for neighbor2 in range(neighbor+1, len(neighbors[eliminationkey])):
                        listforval = neighbors.get(eliminationkey)
                        value1 = listforval[neighbor]
                        value2 = listforval[neighbor2]
                        if originalgraph.has_edge(value1, value2) == False:
                            originalgraph.add_edge(value1, value2)
                set_var.remove(eliminationkey)
                #nx.draw(originalgraph, with_labels=True, node_size=3000)
                #plt.show()
                print('set_var =' , set_var)
                print('edges = ', originaledges)
            print(ordering)
            return ordering          
        # return good ordering for elimination of set_var based on min-degree heuristics and min-fill heuristics
    
    def Variable_Elimination(self, x):
            cpt = self.bn.get_cpt(x)
            old_length = int(len(cpt.index))

            while old_length > 2:
                row_names = cpt.index.values
                steps = old_length/2
                col_names = cpt.columns
                first_col = col_names[0]
                #print(row_names)
                for index in range(0, int(old_length/2)):
                    cpt['p'][index] = cpt['p'][index] + cpt['p'][index+steps]      
                cpt.drop(first_col, inplace = True, axis=1)
                
                uneven_indexes = list(range(int(old_length/2),old_length))
                #print(uneven_indexes)
                for index in uneven_indexes:
                    cpt = cpt.drop(row_names[index], axis=0)
                old_length = int(len(cpt.index))
            return(cpt)
    # return input_net without set_var
    
    def Marginal_Distributions(self, q, e): #Basis staat maar is nog niet af. gaat mis bij marginalization / summing out. moet alles 
        e_index = e.index.tolist()
        all_cpts = self.bn.get_all_cpts()
        print(e_index)
        #Reduce all factors with regard to e)
        for cpt in all_cpts:
            Reduced_factors = self.Network_Pruning(q, e)
         #lijkt goed te gaan???
        # variable elimination / marginalization of (q and e)
        varstoeliminate = q.append(e_index)
        print(q, varstoeliminate)
        print('varstoeliminate =', varstoeliminate)
        ordering = self.Ordering(varstoeliminate, 'min-fill')
        print('ordering = ', ordering)
        for i in ordering:
            self.Marginalization(i)
        joint_marginal = self.bn.get_all_cpts()
        print('joint marginal =', joint_marginal)
        # Sum out Q to get Pr(Q|e)
        posterior_marg = joint_marginal / e
        print(posterior_marg)
        cpt = self.bn.get_cpt(q)
        return cpt['p']
        # return marginal distribution P(q|e)
        
    def MAP(self, q, e):
        MAP = []
        # prune net based on q and e
        self.Network_Pruning(q, e)

        #sum out q from net
        for i in q:
            self.Marginalization(i)
        
        vars = self.bn.get_all_variables()
        #get elimination order from ordering
        ordering = self.Ordering(vars, "min-degree")
        
        #for each var: get max prob. and assign that truth value and update with this prob. factor 
        for var in ordering:  
            #get max f in cpt                  
            cpt = self.bn.get_cpt(var)
            
            #true max  
            cpt_true = cpt.loc[cpt[var] == True]         
            max_true = cpt_true.max()
            max_p_true = max_true['p']            
            
            #false max            
            cpt_false = cpt.loc[cpt[var] == False]         
            max_false = cpt_false.max()
            max_p_false = max_false['p']            
            
            #set this truth value and append to MAP
            if max_p_true >= max_p_false:
                truth_value = True
            else:
                truth_value = False
                
            assignment = pd.Series({var : truth_value})
            MAP.append(assignment)    
            
            #update child cpts
            children = self.bn.get_children(var)
            for child in children:
                    
                print(f"Parent cpt: \n {self.bn.get_cpt(var)}")
                # print(f"Child cpt: \n {self.bn.get_cpt(child)}")
                    
                #apply factor multiplication with max true and max false 
                cpt_child= self.bn.get_cpt(child)
                
                cpt_rows_true = cpt_child.loc[cpt_child[var] == True]
                rows_true = cpt_rows_true.index.values
                cpt_rows_false = cpt_child.loc[cpt_child[var] == False]
                rows_false = cpt_rows_false.index.values
                
                for row in rows_true:
                    cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], [max_p_true])
                
                for row in rows_false:
                    cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], [max_p_false])
                    
                # print(f"Child cpt after multiplication: \n {self.bn.get_cpt(child)}")
                    
                #max-out var from children
                new_cpt = self.Maxing_Out(child)
                if child in new_cpt.columns:
                    new_cpt = new_cpt.drop(child, axis=1)                
                # print(f"Child cpt after maxing out: \n {self.bn.get_cpt(child)}")
                
                #fix index values
                length = len(new_cpt.index.values)
                new_indexes = []
                for i in range(0,length):
                    new_indexes.append(i)
                new_cpt.index = [new_indexes]
                self.bn.update_cpt(child, new_cpt)
                # print(f"Child cpt after fixing row indexes: \n {self.bn.get_cpt(child)}")
                                
        return MAP
        # return most probable explanation given e
    
    def MPE(self, e):
        MPE = []
        #first apply network pruning given evidence e
        vars = self.bn.get_all_variables()
        vars.remove(e.index)#only works for 1 evidence var
        self.Network_Pruning(vars, e)
        
        #get elimination order from ordering
        ordering = self.Ordering(vars, "min-degree")
        
        #for each var: get max prob. and assign that truth value and update with this prob. factor 
        for var in ordering:  
            #get max f in cpt                  
            cpt = self.bn.get_cpt(var)
            
            #true max  
            cpt_true = cpt.loc[cpt[var] == True]         
            max_true = cpt_true.max()
            max_p_true = max_true['p']            
            
            #false max            
            cpt_false = cpt.loc[cpt[var] == False]         
            max_false = cpt_false.max()
            max_p_false = max_false['p']            
            
            #set this truth value and append to MPE
            if max_p_true >= max_p_false:
                truth_value = True
            else:
                truth_value = False
                
            assignment = pd.Series({var : truth_value})
            MPE.append(assignment)    
            
            #update child cpts
            children = self.bn.get_children(var)
            for child in children:
                    
                print(f"Parent cpt: \n {self.bn.get_cpt(var)}")
                # print(f"Child cpt: \n {self.bn.get_cpt(child)}")
                    
                #apply factor multiplication with max true and max false 
                cpt_child= self.bn.get_cpt(child)
                
                cpt_rows_true = cpt_child.loc[cpt_child[var] == True]
                rows_true = cpt_rows_true.index.values
                cpt_rows_false = cpt_child.loc[cpt_child[var] == False]
                rows_false = cpt_rows_false.index.values
                
                for row in rows_true:
                    cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], [max_p_true])
                
                for row in rows_false:
                    cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], [max_p_false])
                    
                # print(f"Child cpt after multiplication: \n {self.bn.get_cpt(child)}")
                    
                #max-out var from children
                new_cpt = self.Maxing_Out(child)
                if child in new_cpt.columns:
                    new_cpt = new_cpt.drop(child, axis=1)                
                # print(f"Child cpt after maxing out: \n {self.bn.get_cpt(child)}")
                
                #fix index values
                length = len(new_cpt.index.values)
                new_indexes = []
                for i in range(0,length):
                    new_indexes.append(i)
                new_cpt.index = [new_indexes]
                self.bn.update_cpt(child, new_cpt)
                # print(f"Child cpt after fixing row indexes: \n {self.bn.get_cpt(child)}")
                                
        return MPE
        # return most probable explanation given e

class main():
    # Variables
    Truth_value = True    
    
    Query_var = ["dog-out"]
    Var = "family-out"
    
    # Query_var = ['Rain?', 'Wet Grass?']
    # Var = "Sprinkler?"
    Evidence = pd.Series({Var : Truth_value})
    
    x = ['bowel-problem', 'dog-out']
    y = ['hear-bark']
    z = ['family-out']
    
    #Init net
    NET = BNReasoner("testing/dog_problem.BIFXML") #initializing network)
    # NET = BNReasoner("testing/lecture_example.BIFXML") #initializing network)   
    
    #Applying marginalization    
    # Var2 = 'hear-bark'
    # cpt = NET.Get_CPT(Var2)
    # f = cpt['p']   
    # NET.Marginalization(f, Var2)
    
    #Applying MPE
    #NET.MAP(x, Evidence)      
    
    # Applying marginal distributions
    # NET.Marginal_Distributions(x, Evidence)
    # vars = NET.bn.get_all_variables()
    # for var in vars:
    #     print(f"{NET.bn.get_cpt(var)}\n")
    
    #print(NET.MPE(Evidence))
    #print(NET.MAP(z, Evidence))
    
    #finding a good ordering for variable elimination
    # NET.Ordering(NET.Get_Vars())
    
if __name__ == "__main__":
    main()