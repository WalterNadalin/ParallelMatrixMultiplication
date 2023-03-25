from matplotlib.pyplot import savefig, subplots, show, title, suptitle
from numpy import array, zeros, arange

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
        
with open('data/times.txt') as f:
    data = [[num(x) for x in line.split()[1:]] for line in f.readlines()]

nodes = 4
section = 11
naive = data[nodes * (section - 1):nodes * section]
dgemm = data[nodes * section:nodes * (section + 1)]
processors = array([n for _, n, *_ in naive]) // 32
naive_comm = array([t for *_, t, _ in naive])
naive_comp = array([t for *_, t in naive])
dgemm_comm = array([t for *_, t, _ in dgemm])
dgemm_comp = array([t for *_, t in dgemm])
n = naive[0][0]

measures = ((naive_comm, naive_comp), (dgemm_comm, dgemm_comp))

fig, ax = subplots(layout='constrained')
bottom = zeros(nodes)
width = 0.2
multiplier = 0
i = 0
bar_colors = ('black', 'red',  'black', 'yellow')
label = ("Communication", "Naivë computation", None, "DGEMM computation")

for times in measures:
    offset = width * multiplier - width / 2
    
    for time in times:
        p = ax.bar(processors + offset, time, width, label = label[i], bottom = bottom, \
                   color = bar_colors[i], edgecolor = 'black')
        bottom += times[0] 
        i += 1

    multiplier += 1
    bottom = zeros(nodes)

ax.set_xticks(processors)
ax.set_ylabel(r"Time [$s$]")
ax.set_xlabel("Number of nodes")
ax.legend(loc = "upper right")
ax.grid(linestyle = '--', axis = 'y')

title(f"Size of matrices: {n}"+r"$\times$"+f"{n}", fontsize = 10)
suptitle('Communication and computation times per number of nodes', fontsize = 13, y = 1.03, x = 0.54)
savefig(f'analysis/{n}.png')
show()

