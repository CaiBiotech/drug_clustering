# -*- coding:utf-8 -*-

__cmddoc__ = """

easy_DBCclustering.py - Reads an SDFile and clustering molecules with Butina Clustering Algorithm using RDKIT tools
#Contact: haiming_cai@hotmail.com - 2022 - CHINA-VMI

Basic usage: 
python easy_clustering.py -in TEST.sdf
python easy_clustering.py -in TEST.smi
python easy_clustering.py -in TEST.smi -m Taylor-Butina
python easy_clustering.py -in TEST.smi -m Butina

""" 
 
## 定义加载包
import argparse
import logging
import pprint
import sys
import os
import subprocess as sp
from io import StringIO
import pandas as pd
import numpy as np
from rdkit import RDLogger
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import pickle as pic
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import chemfp
from chemfp import search

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

#import warnings
#warnings.filterwarnings("Error")
#warnings.filterwarnings("ignore")
#sio = sys.stderr = StringIO()

def sdf2smidf(sdffile, key1="ID"):
    mylogger = RDLogger.logger()
    mylogger.setLevel(val=RDLogger.ERROR)

    suppl = [ mol for mol in Chem.SDMolSupplier(sdffile) ]
    ID_list = []
    smiles_list = []
    for mol in suppl:
        if mol:
           name = mol.GetProp(key1)
           ID_list.append(name)
           try:
              smi = Chem.MolToSmiles(mol,isomericSmiles=False)
              smiles_list.append(smi)
           except:
              pass
    mol_datasets = pd.DataFrame({'smiles':smiles_list,'ID':ID_list})
    mol_datasets = mol_datasets[['smiles','ID']]
    #print(mol_datasets)
    return mol_datasets

def smi2smidf(smifile):
    #mol_datasets = pd.DataFrame({'smiles': smis, 'id': ['s' + str(x) for x in range(1, len(smis)+1)]}, columns = ['smiles','id'])
    mol_datasets = pd.read_csv(smifile, sep='\t', names = ['smiles','ID'], encoding='gbk', header = None)
    #print(mol_datasets)
    return mol_datasets

def smidf2csv(smidf, output="smidf.smi"):
    smidf.to_csv(output, header=False, sep ='\t', index=False)

def smidf2fps(smilist, radius, nBits, output='result.fps'):
    mylogger = RDLogger.logger()
    mylogger.setLevel(val=RDLogger.ERROR)

    #out_file = open(output, "w")
    name_list = []
    fps_list = []
    #print(smilist)
    for index,value in smilist.iterrows():
        name = value['ID']
        #print(name)
        smiles = value['smiles']
        #print(smiles)
        name_list.append(name)
        mol = Chem.MolFromSmiles(smiles)
        #print(mol)
        if mol:
           #Morgan Fingerprints摩根指纹
           fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) 
           fps_list.append(fps)
           #print(fps)
           #out_file.write("{}\t{}\n".format(name,fps.ToBitString()))
    fps_datasets = pd.DataFrame({'ID':name_list,'FPS':fps_list})
    #out_file.close()
    return fps_datasets

def ClusterFps(fpslist,cutoff):
    # first generate the distance matrix:
    dists = []
    nfps = len(fpslist)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fpslist[i],fpslist[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    mol_clusters = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    #print_mem(stage='Dist & Clust usage')
    #del dists
    return mol_clusters

def savecluster1(clusterlist, mollist, output='cluster.pic', output2='cluster.csv'):
    with open(output,'wb') as f: pic.dump(cs,f)

    is_centroid = np.zeros(len(mollist)).astype(int)
    clust_idx = np.zeros(len(mollist)).astype(int)

    for idx,C in enumerate(clusterlist):
        is_centroid[C[0]]=1
        clust_idx[list(C)]=idx

    df = pd.DataFrame(dict(clust_id=clust_idx, is_centroid=is_centroid, smiles=mollist))
    df = df.assign(temp = df.clust_id*10-df.is_centroid)
    df.sort_values('temp', inplace=True)
    df = df.drop('temp', 1)
    df.to_csv(output2, sep=';', index=False)

def savecluster2(clusterlist, mollist,outpath='./', outname='id_cluster'):
    nfps = len(mollist['smiles'])
    cluster_id_list = [0]*nfps
    for idx,cluster in enumerate(clusterlist, 1):
        for member in cluster:
            cluster_id_list[member] = idx

    #df = pd.read_csv("./", sep=" ", names=["SMILES", "Name"])
    df = pd.DataFrame(data=None)
    df['Name'] = mollist['ID']
    df['SMILES'] = mollist['smiles']
    df['Cluster'] = [x for x in cluster_id_list]
    df.sort_values('Cluster', inplace=True)
    df.to_csv(os.path.join(outpath,outname+'_DBC.csv'), index=False)

def savecluster3(clusterlist, mollist,outpath='./', output='id_clusterlist'):
    clusters = np.zeros(len(mollist['smiles']), dtype=np.int32)
    for n in range(0, len(clusterlist)):
        idxs = list(clusterlist[n])
        clusters[idxs] = np.ones(len(idxs)) * n

    clusters = pd.DataFrame(clusters, columns=['clusters'])
    clusters.to_csv(os.path.join(outpath,outname+'_DBC.csv'))

def saveclustersvg(clusterlist, mollist, outpath='cluster_svg', size=(400, 200)): #待修改
    if not os.path.isdir(directory):
        os.mkdir(directory)
    ms = mollist['smiles']
    img_wd, img_ht = size
    for n,c in enumerate(clusterlist, 1):
        idx = c[0]
        mol = Chem.MolFromSmiles(ms[idx])
        #print(mol)

        Chem.Kekulize(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(img_wd, img_ht)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        with open(os.path.join(outpath,"cluster-{}.svg".format(n)), 'w') as img_out: 
             img_out.write(svg)

class ClusterResults(object):
    def __init__(self, true_singletons, false_singletons, clusters):
        self.true_singletons = true_singletons
        self.false_singletons = false_singletons
        self.clusters = clusters

def taylor_butina_cluster(similarity_table):

    centroid_table = sorted(((len(indices), i, indices)
                                 for (i,indices) in enumerate(similarity_table.iter_indices())),
                            reverse=True)

    true_singletons = []
    false_singletons = []
    clusters = []

    seen = set()
    for (size, fp_idx, members) in centroid_table:
        if fp_idx in seen:
            # Can't use a centroid which is already assigned
            continue
        seen.add(fp_idx)

        unassigned = set(members) - seen

        if not unassigned:
            false_singletons.append(fp_idx)
            continue

        clusters.append((fp_idx, unassigned))
        seen.update(unassigned)

    return ClusterResults(true_singletons, false_singletons, clusters)

def smidf2arena(smidf, reorder = True):
    import chemfp

    smidf2csv(smidf, output="smidf.smi")
    
    # Generate fps file
    sp.call(['rdkit2fps', './smidf.smi', '-o', 'smidf.fps'])
    
    ## Load the FPs into an arena
    try:
        arena = chemfp.load_fingerprints('./smidf.fps', reorder = reorder)
    except IOError as err:
        sys.stderr.write("Cannot open fingerprint file: %s" % (err,))
        raise SystemExit(2)
    
    # Remove files
    sp.call(['rm', './smidf.smi', './smidf.fps'])
    #print(len(arena.ids))
    return arena

def chemfp(arena, reorder = True, th = 0.8):

    similarity_table = search.threshold_tanimoto_search_symmetric(arena, threshold = th)
    clus_res = taylor_butina_cluster(similarity_table)
    #print(clus_res.clusters)

    out = []
    cs_sorted = sorted([(len(c[1]), c[1], c[0]) for c in clus_res.clusters], reverse = True)
    for i in range(len(cs_sorted)):
        cl = []
        c = cs_sorted[i]
        cl.append(arena.ids[c[2]]) 
        cl.extend([arena.ids[x] for x in c[1]]) 
        out.append(cl)

    for i in range(len(clus_res.false_singletons)):
        cl = [arena.ids[clus_res.false_singletons[i]]]
        out.append(cl)

    for i in range(len(clus_res.true_singletons)):
        cl = [arena.ids[clus_res.true_singletons[i]]]
        out.append(cl)
    clustlist = out
    clustdf = clus_res
    #print(clustlist)
    #print(len(clustlist))
    #return clustlist, clustdf
    return clustdf

def savecluster4(clusterdf, smidf, arena, outpath='./', outname='id_cluster'):
    cldf = pd.DataFrame(columns = ['Name','Smiles','Centryn','Cluster'])
    for clid in range(len(clusterdf.clusters)):
        cent_id = clusterdf.clusters[clid][0]
        Name = arena.ids[cent_id]
        #print(Name)
        # 获取name对应的smiles序列
        Smiles = smidf.loc[smidf['ID']==Name, 'smiles'].iloc[0] 
        #print(Smiles)
        #print(cent_id)
        line = pd.DataFrame([[Name, Smiles,1, clid+1]], columns = ['Name','Smiles','Centryn','Cluster'])
        cldf = pd.concat([cldf, line]) 
        mems = list(clusterdf.clusters[clid][1])
        for mem in mems:
            Name = arena.ids[mem]
            Smiles = smidf.loc[smidf['ID']==Name, 'smiles'].iloc[0]
            line = pd.DataFrame([[Name, Smiles, 0, clid+1]], columns = ['Name','Smiles','Centryn','Cluster'])
            cldf = pd.concat([cldf, line])
    #cldf = cldf['Name','Smiles','Centryn','Cluster']
    cldf.to_csv(os.path.join(outpath, outname+'_TB.csv'), index=False)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='easy_DBCclustering.py', description='clustering compound with Butina Clustering Algorithm using RDkit.')
	parser.add_argument('-in', '--inputfile', dest='IN', help='input sdf or smi file path')
	parser.add_argument('-out', '--outputpath', dest='OUT', default="./", help='file to write results')
	parser.add_argument('-m', '--method', dest='METHOD', default="Butina", help="Method clustering, which including k-mer, Butina and Taylor-Butina.")
	parser.add_argument('-r', '--radius', dest='RADIUS', default=2, help='option for fps')
	parser.add_argument('-b', '--nBits', dest='NBITS', default=1024, help='option for fps')
	parser.add_argument('-lb', '--longbits', dest='LONGBITS', default=16384, help='option for fps')
	parser.add_argument('-k','--cutoff', dest='CUTOFF', default=0.6, help='set cut off value for clustering')
	parser.add_argument('-f','--flag', dest='FLAG', default="ID", help='set id name for each compound')
	parser.add_argument('-v', '--verbose', default=False, action="store_true", help='verbose output')
	args = parser.parse_args()

	#lg = RDLogger.logger()
	#lg.setLevel(RDLogger.ERROR)
	#lg.setLevel(RDLogger.WARNING)

	if args.IN:
		if '.smi' in args.IN:
			dataset1 = smi2smidf(args.IN)
			logging.info('sdf was converted to smidf done')

		elif '.sdf' in args.IN:
			dataset1 = sdf2smidf(args.IN, args.FLAG)
			logging.info('smi was converted to smidf done')
		else:
			logging.info('The input reference files was set with error filetype. Pleasa correct and resubmit again.')

		if args.METHOD == "Butina":

			dataset2 = smidf2fps(dataset1,args.RADIUS, args.NBITS)
			logging.info('smidf was converted to fps done')

			clustdata = ClusterFps(dataset2['FPS'], args.CUTOFF)
			logging.info('fps clustering was done')

			savecluster2(clustdata, dataset1)
			logging.info('clusters written')

			#saveclustersvg(clustdata, dataset1)
			#logging.info('clusters identical structure written')

		elif args.METHOD == "Taylor-Butina":

			dataset2 = smidf2arena(dataset1)
			logging.info('smidf was converted to fps done')

			clustdata = chemfp(dataset2, th = args.CUTOFF)
			logging.info('fps clustering was done')

			savecluster4(clustdata, dataset1, dataset2)
			logging.info('clusters written')

		logging.info("It was finished!")
	else:
		logging.info("input file is empty!")     