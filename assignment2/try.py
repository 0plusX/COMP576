import matplotlib.pyplot as plt
import matplotlib as mp

x = [1,2,3,4,5,6,7,8,9,10,11]
fig,ax = plt.subplots()
ax.plot(range(len(x)),x,'k',label = 'try')
fig.suptitle('Accuracy with different learning rate',fontsize = 20)
ax.set(xlabel='x-label', ylabel='y-label')
ax.legend(loc = 'lower right',shadow = True)
#plt.show()
fig.savefig('try.png')