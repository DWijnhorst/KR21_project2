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
                length = len(new_cpt.index.values)#fix row indexes after removing rows
                new_indexes = []
                for i in range(0,length):
                    new_indexes.append(i)
                new_cpt.index = [new_indexes]
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
        
        self.bn.draw_structure()
        #return pruned network
    
    def D_separated(self, x,y,z): 
            graph = self.bn.get_interaction_graph()              
            iteration = 0
            while iteration == 0 or leafnodes:
                iteration +=1   
                vars = self.Get_Vars()
                leafnodes = []
                #delete every leaf node W not in x y or z
                for i in vars:
                    if (len(self.bn.get_children(i)) == 0) and (i not in x) and (i not in y) and (i not in z):
                        leafnodes.append(i)
                for leafnode in leafnodes:
                    graph.remove_node(leafnode)
                    self.bn.del_var(leafnode)

                if iteration == 1:#this only has to happen once
                #delete every edge outgoing from nodes in z
                    for var in z:
                        childnodes = self.bn.get_children(var)
                        for child in childnodes:
                            graph.remove_edge(var, child)  
                            self.bn.del_edge([var, child])                                          
            #see if exists a path
            for subx in x:
                for suby in y:
                    if nx.has_path(graph, subx, suby) == False :
                        return True
            return False
        #return D-separated == True / False    
    
    def Independence(self, x, y,z):
        if self.D_separated(x,y,z) == True:
            return True
        else:
            return False
        #return independent == True / False
    
    def Marginalization(self, x): 
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
        #fix row indexes after removing rows
        length = len(cpt.index.values)
        new_indexes = []
        for i in range(0,length):
            new_indexes.append(i)
        cpt.index = [new_indexes]
        self.bn.update_cpt(x, cpt)                
        return cpt
        #return CPT where x is summed-out 
    
    def Maxing_Out(self, x):        
        cpt = self.bn.get_cpt(x)
        print(cpt)
        old_length = int(len(cpt.index))
        row_names = cpt.index.values
        max_indexes = []
        for index in range(0, old_length, 2):
            if cpt['p'][index] > cpt['p'][index+1]:
                max_indexes.append(index)
            else:
                max_indexes.append(index+1)        
        min_indexes = []
        for index in range(0, old_length):
            if index not in max_indexes:
                min_indexes.append(index)        
        for index in min_indexes:
            cpt = cpt.drop(row_names[index], axis=0) 
        #fix row indexes after removing rows
        length = len(cpt.index.values)
        new_indexes = []
        for i in range(0,length):
            new_indexes.append(i)
        cpt.index = [new_indexes]
        self.bn.update_cpt(x, cpt)       
        return cpt           
        # return (CPT where x is maxed-out) , (the instantiation of x which led to the maximized value)
    
    def Factor_Multiplication(self, f, g):       
        h = []             
        for factor1 in f:
            for factor2 in g:
                y = round(factor1*factor2,2)
                h.append(y)
        h = pd.Series(h)       
        return h
        # return h (which is fg) 
           
    def Ordering(self,set_var, heuristic):#heuristic is min-degree or min-fill
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
            # cycle:
            for i in range(len(originalset_var)):
                neighbors = dict()
                fill_value_dict = dict()
                # originaledges = originalgraph.edges()
                for var in set_var:
                    neighbors[var] = list(nx.neighbors(originalgraph, var))
                for i in range(len(set_var)):
                    graph = copy.deepcopy(originalgraph)
                    edges = graph.edges()
                    fill_value = 0
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
                            if graph.has_edge(value1, value2) == False:
                                fill_value +=1  #if not a connection, it needs an additional edge, thus fillvalue+1
                    fill_value_dict.update({set_var[i] :  fill_value})
                    # add key of smallest value of dictionary to the elimination queue
                eliminationkey = min(fill_value_dict, key = fill_value_dict.get)
                ordering.append(eliminationkey)                
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
            return ordering          
        # return good ordering for elimination of set_var based on min-degree heuristics and min-fill heuristics
    
    def Variable_Elimination(self, x, truthvalue = 1):#truthvalue = 0 for False or 1 for True (default)
        #get elimination order                  
        elimination_order = [x]#can use ordering for this
        vars_x = self.bn.get_cpt(x).columns[:1] 
        parents = vars_x
        while len(parents) != 0:
            for var in parents:
                elimination_order.append(var)
                vars_var = self.bn.get_cpt(var).columns[:1]
                parents = vars_var.tolist()
                if var in parents:
                    parents.remove(var)                          
        #apply variable elimination
        if len(elimination_order) != 0:#in case of root node
            length = len(elimination_order)
            for i in range(length-2, -1, -1):#start with lastly added parent (not child (-1))
                cpt_parent = self.bn.get_cpt(elimination_order[i+1])
                cpt_child = self.bn.get_cpt(elimination_order[i])                                            
                #factor multiplication
                cpt_rows_true = cpt_child.loc[cpt_child[elimination_order[i+1]] == True]
                rows_true = cpt_rows_true.index.values                
                #catch tuple bug
                if len(rows_true) != 0:
                    if type(rows_true[0]) is tuple:
                        new_rows_true = []
                        for row in rows_true:
                            row = row[0]
                            new_rows_true.append(row)
                        rows_true =  new_rows_true                
                cpt_rows_false = cpt_child.loc[cpt_child[elimination_order[i+1]] == False]
                rows_false = cpt_rows_false.index.values                
                #catch tuple bug
                if len(rows_false) != 0:
                    if type(rows_false[0]) is tuple:
                        new_rows_false = []
                        for row in rows_false:
                            row = row[0]
                            new_rows_false.append(row) 
                        rows_false = new_rows_false    
                if cpt_parent.loc[cpt_parent[elimination_order[i+1]]==True]['p'].size > 0:                    
                    for row in rows_true:
                        cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], cpt_parent.loc[cpt_parent[elimination_order[i+1]] == True]['p'].tolist())                
                if cpt_parent.loc[cpt_parent[elimination_order[i+1]]==False]['p'].size > 0:
                    for row in rows_false:
                        cpt_child['p'][row] = self.Factor_Multiplication([cpt_child.iloc[row]['p']], cpt_parent.loc[cpt_parent[elimination_order[i+1]] == False]['p'].tolist())
                      
                #sum-out parent from child
                old_length = int(len(cpt_child.index))
                row_names = cpt_child.index.values  
                if len(rows_true) != 0 and len(rows_false) != 0:
                    if len(rows_true) + len(rows_false) == 2:
                        for index in range(0, int(old_length/2), 1):#update factors (sum)
                            cpt_child['p'][index] = cpt_child['p'][index] + cpt_child['p'][index+1]                                                                                 
                        for index in range(int(old_length/2),int(old_length)):#drop the last half rows
                            cpt_child = cpt_child.drop(row_names[index], axis=0)
                    else:
                        for index in range(0, int(old_length/2), 1):#update factors (sum)
                            cpt_child['p'][index] = cpt_child['p'][index] + cpt_child['p'][index+2]                                                                                 
                        for index in range(int(old_length/2),int(old_length)):#drop the last half rows
                            cpt_child = cpt_child.drop(row_names[index], axis=0)
                cpt_child = cpt_child.drop(elimination_order[i+1], axis=1)#drop column of marginalized var 
                self.bn.update_cpt(elimination_order[i],cpt_child)
        #return Pr(x)
        return self.bn.get_cpt(x)['p'][truthvalue]#Pr(x=True)
    
    def Marginal_Distributions(self, q, e):#possible evidence empty
        #reduce factors w.r.t. evidence
        vars = self.bn.get_all_variables()
        for var in vars:
            cpt = self.bn.get_cpt(var)
            column_names = cpt.columns
            evidence_var = e.index.values.tolist()
            if evidence_var[0] in column_names:
                cpt = self.bn.reduce_factor(e, cpt)
                cpt = cpt[cpt['p'] != 0] 
                self.bn.update_cpt(var, cpt)                             
                length = len(cpt.index.values)#fix row indexes after removing rows, hier gaat iets niet goed (,)
                new_indexes = []
                for i in range(0,length):
                    new_indexes.append(i)
                cpt.index = [new_indexes]             
                self.bn.update_cpt(var, cpt)                 
        #compute joint marginal via variable elimination
        PrTrue = self.Variable_Elimination(q, 1)
        PrFalse = self.Variable_Elimination(q, 0)        
        #obtain Pr(evidence)
        cpt_evidence = self.bn.get_cpt(evidence_var[0])
        evidenceTrue = cpt_evidence['p']        
        #compute Pr(q|e) and Pr(-q|e)
        marginalTrue = PrTrue / evidenceTrue
        marginalFalse = PrFalse / evidenceTrue
        return (marginalTrue.tolist()[0], marginalFalse.tolist()[0]) 
        # return marginal distribution P(q|e)
        
    def MAP(self, q, e):
        MAP = []
        vars = self.bn.get_all_variables()
        for var in q:
            vars.remove(var)
        #sum out all variables except those in q
        for var in vars:
            new_cpt = self.Marginalization(var)
            self.bn.update_cpt(var, new_cpt)        
        MAP = self.MPE(q,e)   
        return MAP
        # return most probable explanation given e
    
    def MPE(self, q, e):
        MPE = []
        #first apply network pruning given evidence e
        vars = q
        if len(e) != 0:#possible empty evidence
            if e.index in vars:
                vars.remove(e.index)
            self.Network_Pruning(vars, e)        
        #get elimination order from ordering
        ordering = self.Ordering(vars, "min-fill")        
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
                #max-out var from children
                old_length = int(len(cpt_child.index))
                row_names = cpt_child.index.values  
                max_index_false = 0            
                for index in range(0, int(old_length/2), 1):#max in first half rows
                    if cpt_child['p'][index] > cpt_child['p'][max_index_false]:
                        max_index_false = index
                max_index_true = int(old_length/2)
                for index in range(int(old_length/2), int(old_length), 1):#max in second half rows
                    if cpt_child['p'][index] > cpt_child['p'][max_index_true]:
                        max_index_true = index  
                max_indexes = [max_index_false, max_index_true] 
                min_indexes = []
                for index in range(0, old_length-1):
                    if index not in max_indexes:
                        min_indexes.append(index)        
                for index in min_indexes:
                    cpt_child = cpt_child.drop(row_names[index], axis=0)
                #drop parent column in child cpt
                if var in cpt_child.columns:
                    cpt_child = cpt_child.drop(var, axis=1)  
                #fix index values
                length = len(cpt_child.index.values)
                new_indexes = []
                for i in range(0,length):
                    new_indexes.append(i)
                cpt_child.index = [new_indexes]
                self.bn.update_cpt(child, cpt_child)
        return MPE
        # return most probable explanation given e

class main():
    #Init net
    NET = BNReasoner("testing/dog_problem.BIFXML") #initializing network 1  
    NET2 = BNReasoner("testing/lecture_example.BIFXML") #initializing network 2
    
    #testing --> uncomment the function you want to test
    # NET.Network_Pruning(["dog-out"], pd.Series({"family-out" : True}) ) 
    # print(NET.D_separated(['family-out'], ['hear-bark'], ['dog-out']))
    # print(NET.Independence(['family-out'], ['hear-bark'], ['dog-out']))
    # print(NET.Marginalization("dog-out"))
    # print(NET.Maxing_Out("dog-out"))
    # print(NET.Factor_Multiplication(NET.Get_CPT('hear-bark')['p'],NET.Get_CPT('light-on')['p']))
    # print(NET.Ordering(NET.Get_Vars(), 'min-degree'))#or "min-fill"
    # print(NET2.Variable_Elimination('Slippery Road?'))
    # print(NET2.Marginal_Distributions('Slippery Road?', pd.Series({"Winter?" : True})))#vanaf hier moet nog getest worden
    # print(NET2.MAP(['Slippery Road?', "Sprinkler?"], pd.Series({"Winter?" : True})))
    # print(NET2.MPE(NET2.Get_Vars(),pd.Series({"Winter?" : True})))
    
if __name__ == "__main__":
    main()