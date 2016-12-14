import numpy as np
from itertools import *



a = [0,1,2,3,4]

train = []
val = []

for i in range(len(a)):
	intermediate = []
	if(i == 0):
		j = i+4
	else:	 
		j = (i+4)%5
	
	k = (i+3)%5
	if k < i:
		for ii in range(i,len(a)):
			intermediate.append(ii)

		for ii in range(k):
			intermediate.append(ii)	
		#intermediate.append(k)	
	else:	
		for ii in range(i,k+1):
			intermediate.append(ii)

	print (intermediate)
	print j
	print ' '
	