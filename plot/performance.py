import matplotlib.pyplot as plt
import numpy as np
import matplotlib

"""
matplotlib.rcParams.update({'font.size': 22})


plt.rcdefaults()
#fig, ax = plt.subplots(2, 2, figsize=(10,6))
fig, ax = plt.subplots(2)
labels = ('CPU (4 threads)', 'GPU')
y_pos = np.arange(len(labels))
"""

data = {}
data["cpu_nodnn"] = np.array([264.375, 239.99, 246.008, 289.245, 285.276])
data["gpu_nodnn"] = np.array([322.694, 308.153, 296.267, 307.641, 301.924])
data["cpu_dnn"] = np.array([7639.334, 8587.694, 8519.214, 8475.406, 8480.067])
data["gpu_dnn"] = np.array([1165.812, 1143.143, 1152.552, 1160.63, 1157.875])

keys = [k for k in data.keys()]

# process data
performance = {}
error = {}
for k in keys:
    data[k + "_kHz"] = np.divide(8000000, data[k]) / 1000.0
    data[k + "_time"] = np.divide(data[k], 60.0)

for v in ["_kHz", "_time"]:
    for k in ["nodnn", "dnn"]:
        performance[k + v] = [np.mean(data["cpu_" + k + v]), np.mean(data["gpu_" + k + v])]
        error[k + v] = [np.std(data["cpu_" + k + v]), np.std(data["gpu_" + k + v])]

"""
for idx_1, n1 in enumerate(["nodnn", "dnn"]):
    for idx_2, n2 in enumerate(["kHz", "time"]):

        if n1 == "nodnn":
            ax[idx_1][idx_2].barh(y_pos, performance[n1 + "_" + n2], xerr=error[n1 + "_" + n2], align='center', color="blue", label="no DNN evaluation")
        if n1 == "dnn":
            ax[idx_1][idx_2].barh(y_pos, performance[n1 + "_" + n2], xerr=error[n1 + "_" + n2], align='center', color="orange", label = "DNN evaluation")
        if idx_2 == 0:
            ax[idx_1][idx_2].set_yticklabels(labels)
        else:
            ax[idx_1][idx_2].set_yticklabels([])
            ax[idx_1][idx_2].legend()
        ax[idx_1][idx_2].set_yticks(y_pos)
        ax[idx_1][idx_2].invert_yaxis()  # labels read top-to-bottom
        if n2 == "kHz" and idx_1 == 1:
            ax[idx_1][idx_2].set_xlabel('Processing speed (kHz)')
        if n2 == "time" and idx_1 == 1:
            ax[idx_1][idx_2].set_xlabel('Total time (min)')
"""

"""
ax[0].barh(y_pos, performance["nodnn_kHz"], xerr=error["nodnn_kHz"], align='center', color="blue", label="no DNN evaluation")
ax[0].set_yticklabels(labels)
ax[0].set_yticks(y_pos)
ax[0].invert_yaxis()  # labels read top-to-bottom
ax[0].set_title("Analysis ttH(bb) signal: 8E+06 events, 2.6 GB")
ax[0].set_xlim(0, 1.7*max(performance["nodnn_kHz"]))
ax[0].legend()
#ax[0].set_xlabel('Processing speed (kHz)')


ax[1].barh(y_pos, performance["dnn_kHz"], xerr=error["dnn_kHz"], align='center', color="orange", label="DNN evaluation")
ax[1].set_yticklabels(labels)
ax[1].set_yticks(y_pos)
ax[1].invert_yaxis()  # labels read top-to-bottom
ax[1].set_xlabel('Processing speed (kHz)')
ax[1].set_xlim(0, 1.3*max(performance["dnn_kHz"]))
ax[1].legend()
"""

import matplotlib
params = {
    'text.latex.preamble': [r'\\usepackage{gensymb}'],
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'font.size': 10,
    'text.usetex': False,
    'font.family': 'serif',
    'image.cmap': "CMRmap",
    }

matplotlib.rcParams.update(params)

plt.figure(figsize=(6, 3))
plt.set_cmap('CMRmap')

plt.subplot(2,1,1)
plt.title("W/o DNN evaluation", fontsize=12, y=0.7, x=0.8)

width = 1
xs = width*np.array(range(len(performance["nodnn_kHz"])))

color = plt.cm.hsv(0.6)  
plt.barh(xs, performance["nodnn_kHz"][::-1], xerr=error["nodnn_kHz"][::-1], height=width*0.8, orientation="horizontal", color=color)
# for i in range(len(ms2.values)):
#     plt.text(ms2.values[::-1][i]+10, i, "${0:.1f} \pm {1:.1f}$".format(ms2.values[::-1][i], es2.values[::-1][i]), fontsize=12)
plt.yticks(xs, ["CPU (4t)", "GPU"][::-1], fontsize=12)
plt.xticks(fontsize=12)
#plt.xlabel("Processing speed (kHz)", fontsize=16)
plt.xlim(0,55)
plt.ylim(min(xs)-width*0.5, max(xs)+width*0.5)

plt.subplot(2,1,2)
color = plt.cm.hsv(0.1)  
plt.title("With DNN evaluation", fontsize=12, y=0.7, x=0.8)
plt.barh(xs, performance["dnn_kHz"][::-1], xerr=error["dnn_kHz"][::-1], height=width*0.8, orientation="horizontal", color=color)
# for i in range(len(ms1.values)):
#     plt.text(ms1.values[::-1][i]+10, i, "${0:.1f} \pm {1:.1f}$".format(ms1.values[::-1][i], es1.values[::-1][i]), fontsize=12)
plt.yticks(xs, ["CPU (4t)", "GPU"][::-1], fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Processing speed (kHz)", fontsize=12)
plt.xlim(0,15)
plt.ylim(min(xs)-width*0.5, max(xs)+width*0.5)

plt.tight_layout()
plt.suptitle("ttHbb Analysis: {0:.1E} signal events, {1:.1f} GB".format(8E+06, 2.6), fontsize=14, y=1.01, x=0.54)
plt.savefig("performance_joosep.pdf", bbox_inches="tight")


#plt.title("Analysis ttH(bb) signal: 8E+06 events, 2.6 GB", fontsize="x-large", verticalalignment='bottom')
#plt.tight_layout()
#fig.savefig("performance.pdf")

