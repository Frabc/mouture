import numpy as np #import numpy package & give an short alias international convention

import glob # import glob package
csv = glob.glob("dataset/vid16*.csv")[0]    # define csv as the first doc in data set which name start with videoXX (to be change each time)

import pandas as pd #import panda package give an alias pd (international convention)
df = pd.read_csv(csv,sep=';') # data frame (collection par colonne) to read csv with panda and  ; separator 


class P(): # definition of a novel objet class (Assuming 3D point)
    def __init__(self,x:float,y:float,z:float): # init = constructor create the point ; float = for decimal number (si entier put int)
       
        self.x = x ; self.y = y ; self.z = z              # constructeur de variable self
    def __sub__(A,B): return P(A.x-B.x,A.y-B.y,A.z-B.z)   # def de la soustraction entre points A et B pour utiliser le caractère -
    def __add__(A,B): return P(A.x+B.x,A.y+B.y,A.z+B.z)   # idem addittion pour caractere +
    def scale(A,k:float): return P(k*A.x,k*A.y,k*A.z)     # idem scale pour multiplication d'un vecteur par un scalaire (=nombre)
    def __iter__(A): return iter([A.x,A.y,A.z])           # iteration traiter variable A en ligne
    def cross(A,B): return P( A.y*B.z-A.z*B.y ,           # produit vectoriel, (cross product) calcul les 3 determinants of  matrice points  A & point B
                            -(A.x*B.z-A.z*B.x),
                              A.x*B.y-A.y*B.x )
   

class M():         # definition objet matrice  , Assuming 3x3 matrix                                  
    def  __init__(self,cols=[P(1,0,0),P(0,1,0),P(0,0,1)]): # contructor of variable self, valeur par defaut si on specifie pas les colonnes
        self.cols = cols                                   #def colums of the matrix
    def inv(self):                                         # def matrice inverse
        (a00,a10,a20) = self.cols[0] # donne un nom à chaque element de la colonne 0  
        (a01,a11,a21) = self.cols[1]
        (a02,a12,a22) = self.cols[2]
        
        
        d00=a11*a22-a12*a21; d10=a01*a22-a02*a21; d20=a01*a12-a02*a11  # Cofactors calcul colonne 0
        d01=a10*a22-a12*a20; d11=a00*a22-a02*a20; d21=a00*a12-a02*a10  # colonne1
        d02=a10*a21-a11*a20; d12=a00*a21-a01*a20; d22=a00*a11-a01*a10   #colonne 2
       
     
        det= a00*d00 - a01*d01 + a02*d02   # Determinant along 1st row with determinant
       
        
        return M([                         # return 1/det * T-1 de la comatrice 
            P( d00/det,-d01/det, d02/det),
            P(-d10/det, d11/det,-d12/det),
            P( d20/det,-d21/det, d22/det) ])
    def apply(self,A):                     # multiplication matrice par vecteur
        (a00,a10,a20) = self.cols[0]
        (a01,a11,a21) = self.cols[1]
        (a02,a12,a22) = self.cols[2]
        return P(a00*A.x+a01*A.y+a02*A.z, # return a vector  P
                 a10*A.x+a11*A.y+a12*A.z,
                 a20*A.x+a21*A.y+a22*A.z)
cols = {}                                # construire un collection
for name in ["Tc1","Tc2","Tc3","Ttip"]: # T matrice transformation de C1 C2 ...
    cols[f"{name}.x"]=[]                # fichier excel de 12 colones TC1.x Tc1.y ...Ttip.z
    cols[f"{name}.y"]=[]
    cols[f"{name}.z"]=[]
cols["cast"]=[]                         # + 1 colonne pour projection 2D (cast= projection)
threshold = 1.83        #à déterminer manuellement sur graph
for index,row in df.iterrows():   # iterate rows egrener ligne une par une
    # Inputs 
    c1  = P(row["corner1_x"],row["corner1_y"],row["corner1_z"])
    c2  = P(row["corner2_x"],row["corner2_y"],row["corner2_z"])
    c3  = P(row["corner3_x"],row["corner3_y"],row["corner3_z"])
    tip = P(row["pen_tip_x"],row["pen_tip_y"],row["pen_tip_z"])
    if index==0:                  # pour la ligne 0,  calcul i j et k avec les C1 c2 C3 calcul une seule fois
    #if True:                     # transformation adaptée à chaque point de mesure à la place ligne dessus
        # New base vectors:
        i = (c3-c2).scale(1./15)
        j = (c1-c2).scale(1./9.5)
        k = i.cross(j)           
        
        T = M([i,j,k]).inv()    # Rebase transformation matrix
        
        # new origin
        O=T.apply(c2)
        
    # Rebased points
    (x,y,z) = T.apply(c1)-O  # methode iter de classP cf ci dessus mets les valeurs dans la liste des elements x y z
    cols["Tc1.x"].append(x)
    cols["Tc1.y"].append(y)
    cols["Tc1.z"].append(z)
    (x,y,z) = T.apply(c2)-O
    cols["Tc2.x"].append(x)
    cols["Tc2.y"].append(y)
    cols["Tc2.z"].append(z)
    (x,y,z) = T.apply(c3)-O
    cols["Tc3.x"].append(x)
    cols["Tc3.y"].append(y)
    cols["Tc3.z"].append(z)
    (x,y,z) = T.apply(tip)-O
    cols["Ttip.x"].append(x)
    cols["Ttip.y"].append(y)
    cols["Ttip.z"].append(z)
    
    
    if z < threshold:
        cols["cast"].append( -1 ) #casting de z de tip (dernieère ligne traitee) pour les points sur le papier
    else:
        cols["cast"].append( 10 ) # atributs les point des jumps

for col in cols:   #ajoute les colones au data frame d'origine
    df[col]=cols[col]
    
new_csv=csv.replace('.csv', '_rebased.csv')
new_csv=new_csv.replace('dataset', 'dataset_result')

print('creating',new_csv) 
df.to_csv(new_csv)

import plotly.graph_objects as go # une figure avec plusieurs courbes
fig = go.Figure(data=[           # plotly plus complet que plotly express
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
fig.update_layout(template='plotly_white') # fond en blanc du graph3D
fig.update_layout(scene={"zaxis":{"range":[-2, 5]}})


plotfile = csv.replace(".csv","_rebased.html")
plotfile= plotfile.replace('dataset','dataset_result')
print(f"Creating {plotfile}")
fig.write_html(plotfile)

series = {"x":[],"y":[]}
for index,row in df.iterrows():
    if row["Ttip.z"]<threshold:
        series["x"].append(row["Ttip.x"])
        series["y"].append(row["Ttip.y"])

fig = go.Figure(data=[
    go.Scatter( name="cast", x=series['x'], y=series['y'],
                mode="markers", marker={"size":4})
])
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',)

plotfile = plotfile.replace('3D','2D')
print(f"Creating {plotfile}")
fig.write_html(plotfile)
#fig.write_image(plotfile.replace('.html','.png'))

