#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import random
import pandas as pd
import warnings


# In[1]:


# Given an array which begins with null elements (represented by the keyword "None"), 
# the function extract context returns an array with only the elements which are not 
# null of the array passed as a parameter.

def extract_context(cont):
    
    i = 0
    
    while cont[i] == None:
        
        i += 1
       
    return cont[i:]


'''Relates the context of the chain to the respective row of the
context matrix so the next function can compute the next state
of the chain.
The parameters are the array with the last n elements of the
chain, where n is the greatest length of a context and the context
matrix.'''

def context_row(cont, mat_cont):
    
    m = len(mat_cont[:, 0])

    n = len(cont)
    
    # Starts by the last row

    for j in range(m - 1, -1, -1):
        
        aux = extract_context(mat_cont[j, :])
        
        k = len(aux)
        
        # Tests if the context corresponds to a certain row
        
        if (aux == cont[n - k:]).all(): # with cont[n - k:], it gets the last k elements 
            
            # Returns the row number
            return j

        
'''Computes the next state of the chain, but doesn't return anything.
Its parameters are: transition matrix, current context, context matrix, whole chain,
current index of the chain array and the array of the uniforms(for checking the results).'''

def next_state(P, cont, mat_cont, chain, i, unif_array):
    
    n_states = len(P[0])
    
    # Gets row number
    j = context_row(cont, mat_cont)
    
    u = np.random.uniform()
    
    unif_array[i] = u 
    
    # Uses inverse transform sampling to compute
    # the next state
    if u < P[j, 0]:
        
        chain[i] = 0
        
    else:
    
        for k in range(1, n_states):
        
            if np.sum(P[j, 0:k]) <= u < np.sum(P[j, 0:k + 1]):
            
                chain[i] = k
                
                # Breaks the loop so it doesn't run unecessary
                # tests in the above if statement
                break

                
                
                
                
''' The main function of the code. Its parameters are the length of the chain,
    the transition matrix, the initial context to kick-start the chain and the 
    context matrix.'''

def simulate_CMVL(n, P, cont, mat_cont):
    
    # Length of the initial context
    k1 = len(cont)
    
    # Initiates chain array
    chain = np.empty(n)
    
    chain[:] = np.NaN
    
    # Initiates array of the uniforms
    unif_array = np.empty(n)
    
    unif_array[:] = np.NaN
    
    # Greatest length of a context
    l = len(mat_cont[0, :])
    
    # Adds the initial context to the chain array
    chain[:k1] = cont
    
    for  i in range(k1, n):
        
        # Case in which the current length of the chain is less
        # than the greatest length of a context
        if i + 1 < l:
            
            # Gets the relevant part of the chain
            cont = chain[:i]
            
            # Computes the next state of the chain inside the function,
            # but doesn't return anything
            next_state(P, cont, mat_cont, chain, i, unif_array)
            
        else:
            
            # Gets the last "l" elements of the chain
            cont = chain[i - l:i]
            
            next_state(P, cont, mat_cont, chain, i, unif_array)
            
            
            
    return chain, unif_array


# In[4]:


mat_cont = np.array([[0,0], [1,0], [2,0], [0,1], [1, 1], [2,1], [None,2]])

P = np.array([[0,0,1], [0,0,1], [0.2,0.8,0], [0,0,1], [0,0,1], [0.2,0.8,0], [0.2,0.8,0]])

n = 100

cont = np.array([0, 1])

chain, unif_array = simulate_CMVL(n, P, cont, mat_cont)


# In[5]:


chain


# In[6]:


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


# In[8]:


# CTW algorithm
class CTW:
    def __init__(self, depth, symbols, seq, beta):
        
        # tree depth
        self.D = depth
        
        # number of predicted symbols (for keeping counts)
        self.M = symbols 
        
        # Stores the tree
        self.tree = {}
        
        # Stores the sequence in order to compute 
        # the count vector for each context
        self.seq = seq
        
        # Discards the last element of the sequence since
        # we're interested in the contexts
        seq1 = self.seq[:-1]
        
        n = len(seq1)
        
        i = 0
        
        # Puts the leaves in a matrix
        leaves_matrix = -1 * np.ones((1, self.D))
        
        while i <= n - self.D:
            
            leaves_matrix = np.append(leaves_matrix, np.array([seq1[i: i + self.D]]), 0)
            
            i += 1 
        
        
        leaves_matrix = np.delete(leaves_matrix, 0, 0)
        
        # Removes duplicates
        
        # Rows to remove
        remove = np.array([])
        
        m = len(leaves_matrix)
        
        for j in range(m):
            for k in range(j + 1, m):
                
                if (leaves_matrix[j] == leaves_matrix[k]).all():
                    
                    remove = np.append(remove, k)
        
        
        # Deletes duplicates
        leaves_matrix = np.delete(leaves_matrix, remove.astype(int), 0)
        
        self.leaves_matrix = leaves_matrix
        
        # Creates the root node and therefore the whole m-ary tree 
        self.root = Node(self, np.array([]), None)
        
        
        # Computes the count vectors  
        for i in range(0, self.D + 1):
            for j in range(self.seq.size - i):
            
                self.tree[str(self.seq[j : i + j].astype(np.float))].counts[self.seq[i + j]] += 1
                
                #print(str(self.seq[j : i + j].astype(np.float)), i + j, self.tree[str(self.seq[j : i + j].astype(np.float))].counts)
        
        
        # Computes the mixture probability at the node
        self.Pw_lambda = self.tree[str(np.array([]))].weighted_prob(beta)
        
        
    # Adds a node to the tree
    def add_node(self, node):
        
        self.tree[str(node.context)] = node

    
    # Finds the next node of the next leaf from which we
    # will obtain new children
    def next_leaf(self, leaf_array):
        
        leaf_array += 1
        
        # This bit is so we don't get an index out of range error
        if leaf_array >= len(self.leaves_matrix[:, 0]):
            
            return
                   
            
        i = 1
                
        next_node = self.leaves_matrix[leaf_array, :i]
            
                
        # Finds out what is the next node to be added to the tree from 
        # a new context in the leaves_matrix array
        while str(next_node) in self.tree :
                    
            i += 1
                    
            next_node = self.leaves_matrix[leaf_array, :i]
                
            if i > self.D: break

                
        if i <= self.D:
                
            for j in range(self.M):

                next_context = np.append(next_node[:-1], j)

                # Element of the tree whose children we want to create
                tree_element = self.tree[str(next_node[:-1])]
                
                # Creates the next child
                tree_element.children[str(next_context)] = Node(self, next_context, tree_element, leaf_array)
                
        else: self.next_leaf(leaf_array)
        
        
        
        
        
class Node:
    
    # Leaf_array is the index of the leaf in the leaves_matrix array 
    # to which the current context corresponds
    def __init__(self, ctw, context, parent, leaf_array = 0):
        
        self.ctw = ctw
        
        self.counts = np.zeros((self.ctw.M))
        
        self.context = context
        
        self.parent = parent
        
        #self.beta = 1
        
        if self.context.size == 0:
            
            self.root = True
            
        else:
            
            self.root = False
        
        
        
        self.children = {}
        

        # Add the node to the tree
        self.ctw.add_node(self)
            
        context_len = len(self.context)
        
        # The code inside this if clause does the same as the next's but
        # we still have to consider this case apart
        if self.root:
            
            for j in range(self.ctw.M):
                    
                next_context = np.append(self.context, j)
                    
                self.children[str(next_context)] = Node(ctw, next_context, self, leaf_array)
            
        
        # Checking if the context is not the leaf having
        # length equal to the maximum depth and if the context 
        # is part of the current leaf
        elif context_len < self.ctw.D and self.context[-1] == self.ctw.leaves_matrix[leaf_array, context_len - 1]:
                
            for j in range(self.ctw.M):
                    
                next_context = np.append(self.context, j)
                    
                self.children[str(next_context)] = Node(ctw, next_context, self, leaf_array)
        
        # Case in which we go to the next leaf
        elif (context_len == self.ctw.D and leaf_array < len(self.ctw.leaves_matrix[:, 0]) - 1
        and self.context[-1] == self.ctw.M - 1): # The last condition is to be sure this is the last child of the node
            
            # Calls the recursive method of the CTW class
            self.ctw.next_leaf(leaf_array)
            
            
            
            
        if self.children == {}:
            
            self.leaf = True
                
        else: 
            
            self.leaf = False
        
        
    def estimated_prob(self):
        
        m = self.ctw.M
        
        M_s = self.counts.sum().astype(int)
        
        # Numerator
        num = 1
        
        # Denominator
        den = 1
        
        for i in range(m):
            
            for j in range(self.counts[i].astype(int)):
                
                num *= (j + 0.5)
                
        for k in range(M_s):
                
            den *= (m/2 + k)

        return num/den
    
    
    def weighted_prob(self, beta):
            
        if self.leaf:
            
            return self.estimated_prob()
        
        else:
            
            Pw_children = 1
            
            for child in self.children:
                
                Pw_children *= self.children[child].weighted_prob(beta)
            
            return beta * self.estimated_prob() + (1 - beta) * Pw_children
        


# In[9]:


with warnings.catch_warnings():
    
    warnings.simplefilter("ignore")
    
    fxn()
    
    seq = np.array([2,0,0,1,2,0,2,0,0,2])
    
    ctw = CTW(depth = 3, symbols = 3, seq = seq, beta = 2/3)


# In[10]:


ctw.tree


# In[12]:


for node in ctw.tree:
    
    print(node, ctw.tree[node].counts, ctw.tree[node].estimated_prob())


# In[13]:


ctw.Pw_lambda

