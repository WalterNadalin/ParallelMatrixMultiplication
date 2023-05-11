from matplotlib.pyplot import savefig, subplots, show, title, suptitle
from numpy import array, zeros, arange

nodes_number = 4

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

lines = 0    
section_witdh = nodes_number * 3
data = []

with open('data/times.txt') as file:	
	for line in file.readlines():
		data += [[num(x) for x in line.split()[1:]]]
		lines += 1

sections = lines // section_witdh

for s in range(sections):
	naive = data[s * section_witdh : nodes_number + s * section_witdh]
	dgemm = data[nodes_number + s * section_witdh : 2 * nodes_number + s * section_witdh]
	cudgemm = data[2 * nodes_number + s * section_witdh : 3 * nodes_number + s * section_witdh]
	n = naive[0][0]
	versions = (naive, dgemm, cudgemm)
	communication = [array([t for *_, t, _ in version]) for version in versions]
	computation = [array([t for *_, t in version]) for version in versions]

	measures = (communication, computation)
	nodes = array([2 ** i for i in range(nodes_number)])

	fig, ax = subplots(layout='constrained')
	bottom = zeros(nodes_number)
	width = 0.2
	multiplier = 0
	i = 0
	bar_colors = ('black', 'black', 'black', 'red', 'yellow', 'green')
	label = ("Communication", None, None, "Naivë computation", "DGEMM computation", "CUDGEMM computation")

	#print(measures)
	for measure_type in measures:
	  
		for times in measure_type:
			if i > 2:
				bottom = measures[0][i - 3]
		  
			offset = width * multiplier - width

			p = ax.bar(nodes + offset, times, width, label = label[i], bottom = bottom, \
						     color = bar_colors[i], edgecolor = 'black')
						     
			i += 1
			multiplier += 1
		
		multiplier = 0
		bottom = zeros(nodes_number)

	ax.set_xticks(nodes)
	ax.set_ylabel(r"Time [$s$]")
	ax.set_xlabel("Number of nodes")
	ax.legend(loc = "upper right")
	ax.grid(linestyle = '--', axis = 'y')

	title(r"$\bf{Communication}$ $\bf{and}$ $\bf{computation}$ $\bf{times}$ $\bf{per}$" + \
		  	r' $\bf{number}$ ' + r'$\bf{of}$ $\bf{nodes}$'+ f"\nSize of matrices: {n}"+r"$\times$"+f"{n}", fontsize = 10)

	savefig(f'analysis/{n}.png')
    # show()

