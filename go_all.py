# Created on Thu Aug  7 13:40:48 2025

#@author: Francoise P
import plot_raw
import glob 
for csv in glob.glob('dataset/*.csv'):
    html=csv.replace('.csv', '.html')
    html=html.replace('dataset', 'dataset_result') 
    print(f'processing {csv}->{html}...')
    plot_raw.makehtml(csv, html)
    

