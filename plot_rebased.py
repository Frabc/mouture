import numpy as np

import glob
csv = glob.glob("dataset/vid13*.csv")[0]

import pandas as pd
df = pd.read_csv(csv,sep=';')

from dataclasses import dataclass

@dataclass
class P(): #Assuming 3D point
    x:float;y:float;z:float;
    def __sub__(A,B): return P(A.x-B.x,A.y-B.y,A.z-B.z)
    def __add__(A,B): return P(A.x+B.x,A.y+B.y,A.z+B.z)
    def scale(A,k:float): return P(k*A.x,k*A.y,k*A.z)
    def __iter__(A): return iter([A.x,A.y,A.z])
    def cross(A,B): return P( A.y*B.z-A.z*B.y ,
                            -(A.x*B.z-A.z*B.x),
                              A.x*B.y-A.y*B.x )

class M(): # Assuming 3x3 matrix
    def  __init__(self,cols=[P(1,0,0),P(0,1,0),P(0,0,1)]):
        self.cols = cols
    def inv(self):
        (a00,a10,a20) = self.cols[0]
        (a01,a11,a21) = self.cols[1]
        (a02,a12,a22) = self.cols[2]
        # Cofactors
        d00=a11*a22-a12*a21; d10=a01*a22-a02*a21; d20=a01*a12-a02*a11
        d01=a10*a22-a12*a20; d11=a00*a22-a02*a20; d21=a00*a12-a02*a10
        d02=a10*a21-a11*a20; d12=a00*a21-a01*a20; d22=a00*a11-a01*a10
        # Determinant along 1st row
        det= a00*d00 - a01*d01 + a02*d02
        # transpose and scale
        return M([
            P( d00/det,-d01/det, d02/det),
            P(-d10/det, d11/det,-d12/det),
            P( d20/det,-d21/det, d22/det) ])
    def apply(self,A):
        (a00,a10,a20) = self.cols[0]
        (a01,a11,a21) = self.cols[1]
        (a02,a12,a22) = self.cols[2]
        return P(a00*A.x+a01*A.y+a02*A.z,
                 a10*A.x+a11*A.y+a12*A.z,
                 a20*A.x+a21*A.y+a22*A.z)
cols = {}
for name in ["Tc1","Tc2","Tc3","Ttip"]:
    cols[f"{name}.x"]=[]
    cols[f"{name}.y"]=[]
    cols[f"{name}.z"]=[]
cols["cast"]=[]
threshold = -95.5
for index,row in df.iterrows():
    # Inputs
    c1  = P(row["corner1_x"],row["corner1_y"],row["corner1_z"])
    c2  = P(row["corner2_x"],row["corner2_y"],row["corner2_z"])
    c3  = P(row["corner3_x"],row["corner3_y"],row["corner3_z"])
    tip = P(row["pen_tip_x"],row["pen_tip_y"],row["pen_tip_z"])
    if index==0:
        # New base vectors:
        i = (c3-c2).scale(1./15)
        j = (c1-c2).scale(1./9)
        k = i.cross(j)
        # Rebase transformation matrix
        T = M([i,j,k]).inv()
    # Rebased points
    (x,y,z) = T.apply(c1)
    cols["Tc1.x"].append(x)
    cols["Tc1.y"].append(y)
    cols["Tc1.z"].append(z)
    (x,y,z) = T.apply(c2)
    cols["Tc2.x"].append(x)
    cols["Tc2.y"].append(y)
    cols["Tc2.z"].append(z)
    (x,y,z) = T.apply(c3)
    cols["Tc3.x"].append(x)
    cols["Tc3.y"].append(y)
    cols["Tc3.z"].append(z)
    (x,y,z) = T.apply(tip)
    cols["Ttip.x"].append(x)
    cols["Ttip.y"].append(y)
    cols["Ttip.z"].append(z)
    if z < threshold:
        cols["cast"].append( -100 )
    else:
        cols["cast"].append( 0 )

for col in cols:
    df[col]=cols[col]
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Scatter3d(   name=f"{name}",
                    x=df[f'{name}.x'],
                    y=df[f'{name}.y'],
                    z=df[f'{name}.z'],marker={"size":1})
    for name in ["Tc1","Tc2","Tc3"]
] + [
    go.Scatter3d(   name="Ttip",
                    x=df['Ttip.x'],
                    y=df['Ttip.y'],
                    z=df['Ttip.z'],marker={"size":2,
                        "color":np.where(df["Ttip.z"]< threshold,"red","rgba(0,0,255,10)")
                    }),
    go.Scatter3d(   name="Cast",
                    x=df['Ttip.x'],
                    y=df['Ttip.y'],
                    z=df['cast'],mode="markers",marker={"size":2, "color": "chocolate"}
                    )

])
fig.update_layout(scene={"zaxis":{"range":[-100, -20]}})

plotfile = csv.replace(".csv","_rebased.html")
plotfile= plotfile.replace('dataset','dataset_result')
print(f"Creating {plotfile}")
fig.write_html(plotfile)
