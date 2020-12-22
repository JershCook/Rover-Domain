#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from random import randint,gauss,shuffle,random,seed


 
 
#np.random.seed(123)
#seed(123)
 
class net:
	def __init__(self,s):
		self.shape=s
		self.depth=len(s)-1

	   
		self.shuffle()
		self.e=0.0
		
	   
		
	def shuffle(self):
		s=self.shape
		self.w=[np.random.normal(size=[s[i],s[i+1]]) for i in range(len(s)-1)]
		self.b=[np.random.normal(size=[1,s[i+1]]) for i in range(len(s)-1)]
 

   
	def cross(self,p,p1,p2):
		for i in range(len(p)):
			P=np.random.random(p[i].shape)<.5
			nP=np.logical_not(P)
			p[i][P]=p1[i][P]
			p[i][nP]=p2[i][nP]
	def copy(self,p):
		for i in range(len(self.w)):
			self.w[i]=p.w[i].copy()
			self.b[i]=p.b[i].copy()
			 
	   
	def crossover(self,p1,p2):
		self.cross(self.w,p1.w,p2.w)
		self.cross(self.b,p1.b,p2.b)

 
	def mut(self,p,m,rad):
		for i in range(len(p)):
			P=np.random.random(p[i].shape) > m
			if(self.bloom<0.95):
				d=np.random.normal(0,rad,p[i].shape)
			else:
				d=np.random.normal(0,1.0/rad,p[i].shape)
			d[P]=0
			p[i]+=d
				
	def mutate(self,mut,rad):
		self.bloom=random()
		self.mut(self.w,mut,rad)
		self.mut(self.b,mut,rad)

		   
	def s(self,x):
		return 1.0/(1.0+np.exp(-x))
   
	def h(self,x):
		return np.tanh(x)
 
	def l(self,x):
		return x
   
	   
	
	def feed(self,x):
		
		for w,b in zip(self.w,self.b):
			x=self.h(np.matmul(x,w)+b)
		return x
 
	def error(self,x,y):
		Y=self.feed(x)
		self.e=np.sum((Y-y)**2)
		return self.e
	



class agent:
	def __init__(self,s,n,env=None,N=.1,L=.1,MUT=.1,RAD=.2):


		
		self.POP=n
		self.NEXT=int(N*n) # elite
		self.LUCK=int(L*n)
		self.CHILDREN=self.POP-self.NEXT-self.LUCK
		self.MUT=MUT
		self.RAD=RAD
		self.environ=env
		
		

		self.pop=[net(s) for i in range(self.POP)]
		
		self.best=self.pop[0]

	def policies(self):
		return [p.feed for p in self.pop]

	def train(self,rewards=None,tournament=True):
	
	
		if self.environ != None:
			for p in self.pop:
				p.e=self.environ(p.feed)	
		else:
			for p,r in zip(self.pop,rewards):
				p.e=r	    	  
		   
		self.pop=sorted(self.pop,key=lambda x: x.e,reverse=True)
	    
		self.best=self.pop[0]
	    
		if tournament:
			mid=self.POP//2
			shuffle(self.pop)
			new=[]

			old1=self.pop[:mid]
			old2=self.pop[mid:]

			for p1,p2 in zip(old1,old2):
				if p1.e > p2.e: 
					best = p1
					worst= p2
				else: 
					best = p2
					worst= p1
				worst.copy(best)
				worst.mutate(self.MUT,self.RAD)

				new.append(best)
				new.append(worst)
				

		else:
			new=self.pop[:self.NEXT]
			old=self.pop[self.NEXT:]
		
			shuffle(old)
			new=new+old[:self.LUCK]
			old=old[self.LUCK:]
			
		
		
			for i in range(len(old)):
				p1,p2=randint(0,self.NEXT+self.LUCK-1),randint(0,self.NEXT+self.LUCK-1)
				Net=old.pop()
				Net.copy(new[p1])
				#Net.crossover(new[p1],new[p1])
				Net.mutate(self.MUT,self.RAD)
				new.append(Net)
				
		self.pop=new
		
		return self.best
		
	def policy(self,index=-1):
		if index<0:
			return self.best
		else:
			return self.pop[index]
	
