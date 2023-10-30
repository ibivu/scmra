import scmra
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import networkx as nx

#read in data
dataWT = pd.read_csv("./test/data/scPopulation_1.tsv", sep='\t')
dataMUT = pd.read_csv("./test/data/scPopulation_2.tsv", sep='\t')
dataWT.index = ["CTR."+str(i) for i in range(dataWT.shape[0])]
dataMUT.index = ["MUT."+str(i) for i in range(dataMUT.shape[0])]
populations = [dataWT, dataMUT]
labels = ["Ctr", "Mut"]
dataScaled, groupAnnot = scmra.prepare_data(populations, labels)
rglob = dataScaled.loc[[n for n in list(dataScaled.index) if ("Active" in n)]]
rglob.index = [i.replace("Active", "") for i in rglob.index]
rtot = dataScaled.loc[[n for n in list(dataScaled.index) if ("Tot" in n)]]
rtot.index = [i.replace("Tot", "") for i in rtot.index]

#make MRA simulation
eta = 0.02
theta = 0.1
scd = scmra.ScData(rglob=rglob, rtot = rtot, group_annot= groupAnnot)
p = scmra.ScCnrProblem(scd, eta=eta, theta=theta) 
p.cpx.write("./test/data/CplexCnrProblem.lp")
p.cpx.solve()
p = None