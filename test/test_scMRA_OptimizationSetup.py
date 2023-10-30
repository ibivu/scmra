import scmra
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import networkx as nx

#read in data
data = pd.read_csv("./test/data/scPopulation_1.tsv", sep='\t')

#PREPARE DATA
data.index = ["ID."+str(i) for i in range(data.shape[0])]
dat_scaled = ((data - data.median())/data.median()).transpose()
rglob = dat_scaled.loc[[n for n in list(dat_scaled.index) if ("Active" in n)]]
rglob.index = [i.replace("Active", "") for i in rglob.index]
rtot = dat_scaled.loc[[n for n in list(dat_scaled.index) if ("Tot" in n)]]
rtot.index = [i.replace("Tot", "") for i in rtot.index]

#make MRA simulation
eta = 0.1
scd = scmra.ScData(rglob=rglob, rtot = rtot)
p = scmra.ScMraProblem(scd, eta=eta)
p.cpx.write("./test/data/CplexMraProblem.lp")
p.cpx.solve()
p = None
