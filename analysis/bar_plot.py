from matplotlib.pyplot import subplots, show, title, suptitle
from numpy import array, zeros, arange

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
        
with open('data/times.txt') as f:
    data = [[num(x) for x in line.split()] for line in f.readlines()]
    
naive = data[:5]
dgemm = data[5:]
processors = array([n for _, n, *_ in naive])
comm = array([t for *_, t, _ in naive])
naive_comp = array([t for *_, t in naive]) / 15 + 1
dgemm_comp = array([t for *_, t in dgemm]) / 15 + 1
n = naive[0][0]

measures = ((comm, naive_comp), (comm, dgemm_comp))

fig, ax = subplots(layout='constrained')
bottom = zeros(5)
width = 0.75
multiplier = 0
i = 0
bar_colors = ('black', 'red',  'black', 'green')
label = ("Communication", "Naive computation", None, "DGEMM computation")

for times in measures:
    offset = width * multiplier - width / 2
    
    for time in times:
        p = ax.bar(processors + offset, time - bottom, width, label = label[i], bottom = bottom, \
                   color = bar_colors[i], edgecolor = 'black')
        bottom += comm 
        i += 1

    multiplier += 1
    bottom = zeros(5)

ax.set_xticks(processors)
ax.set_ylabel(r"Time [$s$]")
ax.set_xlabel("Number of nodes")
ax.legend(loc = "upper right")
ax.grid(linestyle = '--', axis = 'y')

title(f"Size of matrices: {n}"+r"$\times$"+f"{n}", fontsize = 10)
suptitle('Communication and computation times per number of nodes', fontsize = 13, y = 1.03, x = 0.54)
show()
