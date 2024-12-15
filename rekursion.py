import matplotlib.pyplot as plt
import math

def rekursion(p, n):
    if n != 0:
        if n % 2 == 0:
            p = p + 1/(n+1)
        else:    
            p = p - 1/(n+1)
    else:
        return p
    
    print(p)
    plt.scatter(-n, p, color='blue')
    if n > 0:
        return rekursion(p, n-1)
        
    else:
        return p


p = rekursion(9, 100)
print(p)
rekursion(p, 100) 

#plt.yscale('log')
plt.xlabel('n')
plt.ylabel('p')
plt.title('Rekursion Plot')
plt.show()