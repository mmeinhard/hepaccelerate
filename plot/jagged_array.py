from hepaccelerate.utils import NanoAODDataset, choose_backend, JaggedStruct
from lib_analysis import chunks
NUMPY_LIB, ha = choose_backend(False)
NanoAODDataset.numpy_lib = NUMPY_LIB
import os
import hepaccelerate

import matplotlib
fig_width = 7
fig_height = 4
params = {
    'text.latex.preamble': [r'\\usepackage{gensymb}'],
    'axes.labelsize': 8, 
    'axes.titlesize': 8,
    'font.size': 10, 
    'text.usetex': False,
    'figure.figsize': [fig_width,fig_height],
    'font.family': 'serif',
    'image.cmap': "CMRmap",
    }
    
matplotlib.rcParams.update(params)

jets= JaggedStruct.load('test.npz', NUMPY_LIB)

good_jets = NUMPY_LIB.ones(jets.pt.shape[0], dtype = bool)
mask_events = NUMPY_LIB.ones(jets.offsets.shape[0], dtype = bool)
njets = ha.sum_in_offsets(jets, good_jets, mask_events, jets.masks["all"], NUMPY_LIB.int8)[0:50]
print(njets)

var = {}
for f in ["pt", "eta", "phi"]:
    l = []
    for i in range(10):
        idx = NUMPY_LIB.full(mask_events.shape[0], i, dtype=NUMPY_LIB.int8)
        l.append(ha.get_in_offsets(getattr(jets, f), getattr(jets, "offsets"), idx, mask_events, good_jets)[0:50])
        print(l[i][0])
    var[f] = NUMPY_LIB.column_stack(tuple(l))

var["njets"] = NUMPY_LIB.column_stack((njets, njets, njets, njets, njets, njets, njets, njets, njets, njets))

print(var["njets"].shape)

import matplotlib.pyplot as plt
#plt.imshow(var["njets"], cmap="Reds", vmin=2, vmax=12)
#plt.colorbar()
#plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,ncols=4)
#gs1 = matplotlib.gridspec.GridSpec(6, 3)
#gs1.update(wspace=0.025, hspace=0.05)

p1 = ax1.imshow(var["njets"], cmap="Reds", vmin=2, vmax=12)
plt.colorbar(p1,ax=ax1)
ax1.set_ylabel("Event index")
ax1.set_title(r"$N_{jets}$")

p2 = ax2.imshow(var["pt"], cmap="OrRd", vmin=0, vmax=125)
plt.colorbar(p2,ax=ax2, shrink=0.7)
ax2.set_title(r"$p_T$")

p3 = ax3.imshow(var["eta"], cmap="RdYlBu_r", vmin=-4.5, vmax=4.5)
plt.colorbar(p3,ax=ax3, shrink=0.7)
ax3.set_title(r"$\eta$")

p4 = ax4.imshow(var["phi"], cmap="RdYlBu_r", vmin=-3.5, vmax=3.5)
plt.colorbar(p4,ax=ax4, shrink=0.7)
ax4.set_title(r"$\phi$")

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks([])

for ax in [ax2, ax3, ax4]:
    ax.set_yticks([])
    ax.set_xlabel(r"jet index")

#plt.tight_layout()
plt.suptitle(r"Jagged ttH(bb) event content: 50 events, up to 10 jets", fontsize=14, y=1.01, x=0.54)
plt.savefig("ttHbb_jagged_array.pdf", bbox_inches="tight")
#filename = ["root://t3dcachedb.psi.ch//pnfs/psi.ch/cms/trivcat/store/mc/RunIIFall17NanoAOD/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/60000/A0BF520F-4569-E811-9D3E-001E67F8F727.root"] 

#arrays_objects = ["Jet_pt", "Jet_eta", "Jet_phi"]

#dataset = NanoAODDataset(filename, arrays_objects, "Events", ["Jet"], [])
#dataset.get_cache_dir = lambda fn,loc=os.path.join(os.getcwd(), 'cache'): os.path.join(loc, fn)
#dataset.from_cache(verbose=True, nthreads=4)

